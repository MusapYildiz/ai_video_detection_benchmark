"""
IvyFake (Ivy-xDetector) - inference script
---------------------------------------------
Repo (Pi3AI/IvyFake) sadece proje sayfasi (HTML/JS) - gercek model kodu yok.
Bu script, README'deki GORUNTU-odakli ornek kodun VIDEO girisine uyarlanmis hali.

Model: AI-Safeguard/Ivy-Fake (Qwen2.5-VL tabanli)
Cikti formati: <think>...</think> ... <conclusion>real/fake</conclusion>

V100 icin:
    torch_dtype = torch.float16      (orijinal README'de bfloat16 + flash_attention_2 var,
    attn_implementation = "sdpa"      V100'de ikisi de calismaz/yavas calisir)

Kullanim:
    python infer.py --model_path <path> --video_path /path/to/video.mp4
    python infer.py --model_path <path> --video_dir /path/to/videos/ --output results.csv
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from model_stats_utils import get_param_count, get_dir_size_gb, compute_fake_probability


# README'deki system prompt - image icin yazilmis ama video'ya da uygulanabilir
SYSTEM_PROMPT = (
    "You are an AI-generated content detector. Classify the media as real or fake. "
    "Provide reasoning inside <think>...</think> tags. End with exactly one word—real or fake—"
    "wrapped in <conclusion>...</conclusion>."
)
USER_PROMPT_VIDEO = "Is this video real or fake?"


def load_model(model_path):
    print(f"[info] Loading Ivy-xDetector from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,      # V100: bfloat16 degil
        attn_implementation="sdpa",      # V100: flash_attention_2 degil
    )

    # --- ONEMLI DUZELTME ---
    # Bu checkpoint'in ust seviye config.json'unda "tie_word_embeddings" alani
    # eksik (None), ama text_config.tie_word_embeddings=True. transformers'in
    # otomatik tie_weights() mekanizmasi UST SEVIYE config'e bakiyor ve bu
    # multimodal yapida tying'i TETIKLEMIYOR -> lm_head rastgele agirliklarla
    # baslatiliyor -> model coplu/anlamsiz metin uretiyor.
    # Cozum: embed_tokens agirligini lm_head'e elle ata (gercek tying).
    model.lm_head.weight = model.get_input_embeddings().weight

    # Model kucuk (3B, ~6GB fp16) - device_map="auto" gereksiz ve tying'le
    # cakisma riski tasiyor (agirliklar farkli cihazlara dagilabilir).
    # Tek GPU'ya elle tasiyoruz.
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print("[info] Model loaded (lm_head manually tied to embed_tokens).")
    return model, processor


def parse_conclusion(response_text):
    match = re.search(r"<conclusion>(.*?)</conclusion>", response_text, re.DOTALL)
    if match:
        verdict = match.group(1).strip().lower()
        if "fake" in verdict:
            return "FAKE"
        if "real" in verdict:
            return "REAL"
        return f"UNKNOWN({verdict})"
    return "UNKNOWN(no_conclusion_tag)"


def run_inference(model, processor, video_path, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path),
                 "max_pixels": 360 * 420, "nframes": 8},
                {"type": "text", "text": USER_PROMPT_VIDEO},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
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

    fake_prob = compute_fake_probability(
        model, processor, base_text=text, generated_text=response_text,
        tag_open="<conclusion>",
        fake_strings=["fake", " fake", "Fake", " Fake"],
        real_strings=["real", " real", "Real", " Real"],
        images=image_inputs, videos=video_inputs,
    )
    return response_text, fake_prob


def main():
    parser = argparse.ArgumentParser(description="IvyFake (Ivy-xDetector) inference")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='results.csv')
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
                response, fake_prob = run_inference(model, processor, vp, max_new_tokens=args.max_new_tokens)
                elapsed = time.perf_counter() - t0
                verdict = parse_conclusion(response)
                fp_str = f"{fake_prob:.4f}" if fake_prob is not None else ""
                rows.append([str(vp), verdict, fp_str, f"{elapsed:.2f}", response])
                print(f"{vp.name:40s} -> {verdict}  (P(fake)={fp_str}, {elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                rows.append([str(vp), "ERROR", "", f"{elapsed:.2f}", str(e)])
                print(f"{vp.name:40s} HATA: {e}  ({elapsed:.2f}s)")
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'verdict', 'fake_probability', 'inference_seconds', 'full_response'])
            writer.writerows(rows)
        print(f"\n[info] Sonuclar kaydedildi: {args.output}")

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        model_info = {
            'model_name': 'IvyFake',
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
        response, fake_prob = run_inference(model, processor, args.video_path, max_new_tokens=args.max_new_tokens)
        verdict = parse_conclusion(response)
        print("\n--- Model cikisi ---")
        print(response)
        print("\n--- Sonuc ---")
        print(f"Verdict: {verdict}  (P(fake)={fake_prob})")


if __name__ == '__main__':
    main()
