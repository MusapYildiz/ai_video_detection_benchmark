"""
Skyra - transformers tabanli inference engine (vLLM YERINE)
--------------------------------------------------------------
V100 (sm_70) GPU'lar guncel vLLM surumlerini (>=0.20) calistiramadigi icin,
bu modul orijinal eval/inference_end2end/model_engine.py ile AYNI arayuzu
(ayni metod adlari, ayni SYSTEM_PROMPT, ayni build_user_prompt / run_inference /
parse_answer davranisi) sunan, ama vLLM degil `transformers` kullanan bir
alternatiftir.

Kullanim: infer.py icindeki
    from model_engine import ModelEngine
satirini
    from model_engine_hf import ModelEngine
ile degistirmek yeterli - infer.py'nin baska hicbir yerine dokunmaya gerek yok.

V100 icin onemli ayarlar:
    torch_dtype = torch.float16     (bfloat16 DEGIL - V100'de tensor-core hizlanmasi yok)
    attn_implementation = "sdpa"    (flash_attention_2 DEGIL - V100 desteklemiyor)
"""

import re

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from model_stats_utils import get_param_count, get_dir_size_gb, compute_fake_probability


# Orijinal model_engine.py'deki SYSTEM_PROMPT ile BIREBIR ayni
SYSTEM_PROMPT = """\nYou are an expert AI video analyst. Your primary task is to review a sequence of video frames and provide a step-by-step analysis of their authenticity.\n\nYou MUST output your entire analysis using the following structure:\n1.  A `<think>...</think>` block containing your detailed reasoning.\n2.  An `<answer>...</answer>` block containing the final, one-word verdict: 'Fake' or 'Real'.\n\nInside the `<think>` block, you MUST:\n1.  Start by briefly describing the overall content of the video frames.\n2.  Follow a detailed, step-by-step \"discovery\" or \"verification\" process.\n3.  When you identify an artifact (or clear a region), you MUST use a valid L3 Category Name from the \"Artifact Category Definitions\" provided below.\n4.  You MUST embed your finding using the following exact tag structure:\n    <type>L3 Category Name</type> in <t>[startTime, endTime]</t> at <bbox>[x1, y1, x2, y2]</bbox>\n5.  If multiple artifacts are present, you must find and tag all of them in temporal order.\n6.  Your entire reasoning process must be self-contained\n\n---\n## Artifact Category Definitions (Valid L3 Categories for the <type> tag)\n\n### 1. Low-Level Forgery\n* **1.1 Texture Anomaly**:\n    * <type>Structure Anomaly</type>\n    * <type>Texture Jittering</type>\n    * <type>Unnatural Blur</type>\n* **1.2 Color and Lighting Anomaly**:\n    * <type>Color Over-saturation</type>\n    * <type>Lighting Inconsistency</type>\n* **1.3 Move Forgery**:\n    * <type>Camera Motion Inconsistency</type>\n\n### 2. Violation of Laws\n* **2.1 Object Inconsistency**:\n    * <type>Abnormal Object Disappearance</type>\n    * <type>Abnormal Object Appearance</type>\n    * <type>Person Identity Inconsistency</type>\n    * <type>General Object Identity Inconsistency</type>\n    * <type>Shape Distortion</type>\n* **2.2 Interaction Inconsistency**:\n    * <type>Abnormal Rigid-Body Crossing</type>\n    * <type>Abnormal Multi-Object Merging</type>\n    * <type>Abnormal Object Splitting</type>\n    * <type>General Interaction Anomaly</type>\n* **2.3 Unnatural Movement**:\n    * <type>Unnatural Human Movement</type>\n    * <type>Unnatural Animal Movement</type>\n    * <type>Unnatural General Object Movement</type>\n* **2.4 Violation of Causality Law**:\n    * <type>Violation of Physical Law</type>\n    * <type>Violation of General Causality Law</type>\n* **2.5 Violation of Commonsense**:\n    * <type>Abnormal Human Body Structure</type>\n    * <type>Abnormal General Object Structure</type>\n    * <type>Text Distortion</type>\n"""


class ModelEngine:
    """transformers tabanli inference engine - orijinal vLLM ModelEngine ile ayni arayuz."""

    def __init__(self, model_path: str, tensor_parallel_size: int | None = None):
        # tensor_parallel_size parametresi imza uyumlulugu icin var, transformers'ta
        # tek GPU kullaniyoruz (device_map='auto' coklu GPU'ya da dagitabilir).
        self.model_path = model_path

        print(f"[ModelEngine-HF] Loading model from {model_path} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,        # V100: bfloat16 degil
            attn_implementation="sdpa",        # V100: flash_attention_2 degil
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("[ModelEngine-HF] Model loaded.")
        self.param_count = get_param_count(self.model)
        self.disk_size_gb = get_dir_size_gb(model_path)
        self._last_fake_prob = None

    def build_user_prompt(self, frame_paths: list[str], timestamps: list[float]) -> str:
        """LADMBench._build_user_prompt ile ayni format."""
        lines = ["Here are the video frames and their corresponding timestamps:"]
        for ts, _ in zip(timestamps, frame_paths):
            lines.append(f"[T={ts:.2f}s] <image>")
        lines.append(
            "\nPlease analyze the video frames, determine if the video is "
            "real or fake, and provide your reasoning."
        )
        return "\n".join(lines)

    def run_inference(self, frame_paths: list[str], user_prompt: str) -> str:
        """Orijinal run_inference ile ayni mesaj/prompt insasi, vLLM yerine transformers.generate()."""
        user_content = []
        frame_path_iter = iter(frame_paths)
        text_parts = re.split(r"<image>", user_prompt)

        for i, text_part in enumerate(text_parts):
            if text_part.strip():
                user_content.append({"type": "text", "text": text_part})
            if i < len(text_parts) - 1:
                try:
                    user_content.append({
                        "type": "image",
                        "image": next(frame_path_iter),
                        "min_pixels": 224 * 224,
                        "max_pixels": 1280 * 28 * 28,
                    })
                except StopIteration:
                    break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,   # orijinaldeki temperature=0 ile esdeger
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response_text = output_text[0]

        fake_prob = compute_fake_probability(
            self.model, self.processor, base_text=prompt, generated_text=response_text,
            tag_open="<answer>",
            fake_strings=["Fake", "fake", " Fake", " fake", "FAKE"],
            real_strings=["Real", "real", " Real", " real", "REAL"],
            images=image_inputs, videos=video_inputs,
        )
        self._last_fake_prob = fake_prob
        return response_text

    def parse_answer(self, response: str) -> str:
        """Orijinal parse_answer ile birebir ayni."""
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if "fake" in answer.lower():
                return "Fake"
            if "real" in answer.lower():
                return "Real"
            return answer
        return "Error"
