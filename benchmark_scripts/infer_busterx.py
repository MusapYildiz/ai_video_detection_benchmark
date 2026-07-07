"""
BusterX (BusterX-plusplus) - transformers tabanli inference (ms-swift/vLLM YERINE)
--------------------------------------------------------------------------------------
Repo'da gercek bir "generate/predict" scripti yok (busterx/eval.py sadece METRIK
hesapliyor, busterx/build_dataset.py sadece veri seti JSON'u hazirliyor). Asil
cikarim ms-swift CLI'sine (`swift infer`) birakilmis. Bu script, busterx/datasetpp/base.py
icindeki varsayilan ayarlari (input_mode="video", sample_fps=2.0, resize=None) ve
prompts/default_user_prompt.txt formatini koruyarak transformers ile dogrudan calisir.

Model: l8cv/BusterX-plusplus (Qwen3.5-4B tabanli, mimari: Qwen3_5ForConditionalGeneration)
NOT: Bu repo'daki ESKI model (l8cv/BusterX_plusplus, alt cizgili) Qwen2.5-VL-7B tabanliydi
ve 2025-07 tarihliydi - KULLANMIYORUZ. Guncel (2026-06) surum Qwen3.5 tabanli, tire'li isim.

V100 icin:
    dtype = torch.float16        (bfloat16 degil)
    attn_implementation = "sdpa" (flash_attention_2 degil)

Kullanim:
    python infer.py --model_path <snapshot_yolu> --video_path /path/to/video.mp4
    python infer.py --model_path <snapshot_yolu> --video_dir /path/to/videos/ --output results.csv
"""

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info

from model_stats_utils import get_param_count, get_dir_size_gb, compute_fake_probability


# prompts/default_user_prompt.txt ile BIREBIR ayni (sadece <video> placeholder'i
# mesaj icerigini olustururken ayriyoruz, metin kismi degismiyor)
USER_PROMPT_TEMPLATE = """<video>
Please analyze whether there are any inconsistencies or obvious signs of forgery in the video, and finally come to a conclusion: Is this video real or fake?
Then, just answer this MCQ with a single letter:
Q: Is this video real or fake?
Options:
A) real
B) fake
Put the final answer in \\boxed{{...}}"""

SAMPLE_FPS = 2.0   # busterx/datasetpp/base.py varsayilani: get_sample_fps() -> 2.0


def load_model(model_path):
    print(f"[info] Loading BusterX-plusplus (Qwen3.5) from {model_path} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,      # V100: bfloat16 degil
        attn_implementation="sdpa",      # V100: flash_attention_2 degil
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("[info] Model loaded.")
    return model, processor


def parse_boxed_answer(response_text):
    """\\boxed{A} / \\boxed{B} formatini parse eder. A=real, B=fake (prompttaki MCQ siralamasi)."""
    match = re.search(r"\\boxed\{(.*?)\}", response_text, re.DOTALL)
    if not match:
        # bazi modeller boxed yerine duz "A)" / "B)" da yazabilir, yedek plan:
        match2 = re.search(r"\b([AB])\)", response_text)
        if match2:
            letter = match2.group(1).upper()
        else:
            return "UNKNOWN", response_text
    else:
        content = match.group(1).strip().upper()
        letter_match = re.search(r"[AB]", content)
        if not letter_match:
            return "UNKNOWN", content
        letter = letter_match.group(0)

    return ("REAL" if letter == "A" else "FAKE"), letter


def run_inference(model, processor, video_path, fps=SAMPLE_FPS, max_new_tokens=2048):
    text_before, text_after = USER_PROMPT_TEMPLATE.split("<video>", 1)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_before},
                {"type": "video", "video": str(video_path), "fps": fps},
                {"type": "text", "text": text_after},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Qwen3.5 ailesi de Qwen3-VL gibi image_patch_size=16 + video_metadata gerektiriyor
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )

    video_metadata = None
    if video_inputs is not None:
        video_inputs, video_metadata = zip(*video_inputs)
        video_inputs, video_metadata = list(video_inputs), list(video_metadata)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response_text = output_text[0]

    # Prompttaki MCQ siralamasi: A) real  B) fake
    fake_prob = compute_fake_probability(
        model, processor, base_text=text, generated_text=response_text,
        tag_open="\\boxed{",
        fake_strings=["B"],
        real_strings=["A"],
        images=image_inputs, videos=video_inputs, video_metadata=video_metadata, **video_kwargs,
    )
    return response_text, fake_prob


def main():
    parser = argparse.ArgumentParser(description="BusterX-plusplus (Qwen3.5) inference - ms-swift/vLLM bypass")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='results.csv')
    parser.add_argument('--fps', type=float, default=SAMPLE_FPS)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    args = parser.parse_args()

    if (args.video_path is None) == (args.video_dir is None):
        sys.exit("Tam olarak bir tanesini ver: --video_path veya --video_dir")

    model, processor = load_model(args.model_path)

    param_count = get_param_count(model)
    disk_size_gb = get_dir_size_gb(args.model_path)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.video_dir is not None:
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_paths = sorted(p for p in Path(args.video_dir).rglob('*') if p.suffix.lower() in video_exts)
        print(f"[info] {len(video_paths)} video bulundu.")

        rows = []
        for vp in video_paths:
            t0 = time.perf_counter()
            try:
                response, fake_prob = run_inference(model, processor, vp, fps=args.fps, max_new_tokens=args.max_new_tokens)
                elapsed = time.perf_counter() - t0
                verdict, raw_letter = parse_boxed_answer(response)
                fp_str = f"{fake_prob:.4f}" if fake_prob is not None else ""
                rows.append([str(vp), verdict, raw_letter, fp_str, f"{elapsed:.2f}", response])
                print(f"{vp.name:40s} -> {verdict}  (P(fake)={fp_str}, {elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                rows.append([str(vp), "ERROR", str(e), "", f"{elapsed:.2f}", ""])
                print(f"{vp.name:40s} HATA: {e}  ({elapsed:.2f}s)")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'verdict', 'raw_letter', 'fake_probability', 'inference_seconds', 'full_response'])
            writer.writerows(rows)
        print(f"\n[info] Sonuclar kaydedildi: {args.output}")

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        model_info = {
            'model_name': 'BusterX-plusplus',
            'param_count': param_count,
            'param_count_billions': round(param_count / 1e9, 2),
            'disk_size_gb': round(disk_size_gb, 2),
            'peak_gpu_memory_gb': round(peak_gb, 2) if peak_gb is not None else None,
        }
        info_path = Path(args.output).with_name('model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"[info] Model bilgisi kaydedildi: {info_path}")
        print(f"[info] {model_info}")

    else:
        response, fake_prob = run_inference(model, processor, args.video_path, fps=args.fps, max_new_tokens=args.max_new_tokens)
        verdict, raw_letter = parse_boxed_answer(response)
        print("\n--- Model cikisi ---")
        print(response)
        print("\n--- Sonuc ---")
        print(f"Verdict: {verdict}  (boxed: '{raw_letter}', P(fake)={fake_prob})")


if __name__ == '__main__':
    main()
