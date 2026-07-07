"""
VideoVeritas - transformers tabanli inference (vLLM YERINE)
---------------------------------------------------------------
Orijinal repo self_scripts/infer/infer_vllm_single.py vLLM + OpenAI-uyumlu API
kullaniyordu. V100 (sm_70) guncel vLLM (>=0.20) calistiramadigi icin bu script
ayni SYSTEM_PROMPT / USER_PROMPT / <answer></answer> parsing mantigini koruyarak
dogrudan `transformers` ile calisir.

NOT: Model mimarisi Qwen2.5-VL DEGIL, Qwen3VL (config.json -> architectures:
['Qwen3VLForConditionalGeneration']). Qwen3-VL'de qwen_vl_utils.process_vision_info()
cagrisina image_patch_size=16 verilmesi ZORUNLU (Qwen2.5-VL'in varsayilani olan
14 ile cagrilirsa token hizalamasi kayar ve cikti anlamsizlasir).

V100 icin:
    dtype = torch.float16        (bfloat16 degil)
    attn_implementation = "sdpa" (flash_attention_2 degil)

Kullanim:
    python infer.py --model_path /path/to/VideoVeritas --video_path /path/to/video.mp4
    python infer.py --model_path /path/to/VideoVeritas --video_dir /path/to/videos/ --output results.csv
"""

import argparse
import csv
import gc
import json
import re
import sys
import time
from pathlib import Path

from model_stats_utils import get_param_count, get_dir_size_gb, compute_fake_probability

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# Orijinal infer_vllm_single.py ile BIREBIR ayni promptlar
SYSTEM_PROMPT = """You are an expert video analyst.
Please think about the question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Finally, give a final verdict within <answer> </answer> tags."""
USER_PROMPT = """Is this video real or fake?"""


def load_model(model_path, device):
    print(f"[info] Loading VideoVeritas (Qwen3VL) from {model_path} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.float16,          # V100: bfloat16 degil
        attn_implementation="sdpa",    # V100: flash_attention_2 degil
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("[info] Model loaded.")
    return model, processor


def parse_verdict(response_text):
    match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if "fake" in answer.lower():
            return "FAKE", answer
        if "real" in answer.lower():
            return "REAL", answer
        return "UNKNOWN", answer
    return "UNKNOWN", response_text


def run_inference(model, processor, video_path, fps=3.0, temperature=0.7, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path), "fps": fps, "max_pixels": 448*448},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Qwen3-VL: image_patch_size=16 SART (Qwen2.5-VL'in varsayilani 14 ile karistirilmamali)
    # return_video_metadata=True SART - aksi halde gercek fps bilinmez, model "fps=24" varsayar
    # ve <t>[start,end]</t> zaman damgalarinin anlami kaybolur.
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )

    video_metadata = None
    if video_inputs is not None:
        # process_vision_info, return_video_metadata=True ile videos listesini
        # (video_tensor, metadata) ciftleri olarak dondurur - ayristirmamiz gerekiyor.
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response_text = output_text[0]

    fake_prob = compute_fake_probability(
        model, processor, base_text=text, generated_text=response_text,
        tag_open="<answer>",
        fake_strings=["fake", "Fake", "FAKE", " fake", " Fake"],
        real_strings=["real", "Real", "REAL", " real", " Real"],
        images=image_inputs, videos=video_inputs, video_metadata=video_metadata, **video_kwargs,
    )
    return response_text, fake_prob


def main():
    parser = argparse.ArgumentParser(description="VideoVeritas (Qwen3VL) inference - vLLM bypass")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, default=None, help="Tek video dosyasi")
    parser.add_argument('--video_dir', type=str, default=None, help="Toplu islenecek video klasoru")
    parser.add_argument('--output', type=str, default='results.csv', help="--video_dir icin cikti CSV")
    parser.add_argument('--fps', type=float, default=3.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    args = parser.parse_args()

    if (args.video_path is None) == (args.video_dir is None):
        sys.exit("Tam olarak bir tanesini ver: --video_path veya --video_dir")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_model(args.model_path, device)

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
                response, fake_prob = run_inference(model, processor, vp, fps=args.fps,
                                                      temperature=args.temperature, max_new_tokens=args.max_new_tokens)
                elapsed = time.perf_counter() - t0
                verdict, raw_answer = parse_verdict(response)
                fp_str = f"{fake_prob:.4f}" if fake_prob is not None else ""
                rows.append([str(vp), verdict, raw_answer, fp_str, f"{elapsed:.2f}", response])
                print(f"{vp.name:40s} -> {verdict}  (P(fake)={fp_str}, {elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                rows.append([str(vp), "ERROR", str(e), "", f"{elapsed:.2f}", ""])
                print(f"{vp.name:40s} HATA: {e}  ({elapsed:.2f}s)")
            finally:
                # ONEMLI: 400+ video uzun bir dongude GPU bellegi birikip OOM'a
                # yol acabiliyor (KV-cache/aktivasyon fragmentasyonu). Her video
                # sonrasi temizlik yapiyoruz.
                gc.collect()
                torch.cuda.empty_cache()

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'verdict', 'raw_answer_tag', 'fake_probability', 'inference_seconds', 'full_response'])
            writer.writerows(rows)
        print(f"\n[info] Sonuclar kaydedildi: {args.output}")

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        model_info = {
            'model_name': 'VideoVeritas',
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
        response, fake_prob = run_inference(model, processor, args.video_path, fps=args.fps,
                                             temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        verdict, raw_answer = parse_verdict(response)
        print("\n--- Model cikisi ---")
        print(response)
        print("\n--- Sonuc ---")
        print(f"Verdict: {verdict}  (raw <answer> icerigi: '{raw_answer}', P(fake)={fake_prob})")


if __name__ == '__main__':
    main()
