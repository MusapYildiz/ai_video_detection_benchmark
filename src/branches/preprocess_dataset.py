import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
preprocess_dataset.py - Videolari onceden isleme
"""

import os
import sys
import csv
import argparse
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import torch

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(THIS_DIR, '..', 'utils')

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, UTILS_DIR)
os.environ['PYTHONPATH'] = THIS_DIR + ':' + UTILS_DIR + ':' + os.environ.get('PYTHONPATH', '')

SKIP_SOURCES = {"zeroscope"}


def process_one(task):
    path, label, source, vtype, out_path, this_dir, utils_dir = task
    import sys, os
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    try:
        from video_io import load_video
        import torch
        bundle = load_video(path, n_frames=16, n_semantic=8, height=224, width=224)
        tensor = bundle.frames_all.half()
        torch.save(tensor, out_path)
        return True, path
    except Exception as e:
        return False, f"{path}: {e}"


def build_index_from_dir(root):
    root = Path(root)
    samples = []
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    for split in ["real", "fake"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        label = 0 if split == "real" else 1
        for subdir in split_dir.iterdir():
            if not subdir.is_dir():
                continue
            source = subdir.name.lower()
            if source in SKIP_SOURCES:
                continue
            for vp in subdir.rglob("*"):
                if vp.is_file() and vp.suffix.lower() in VIDEO_EXTENSIONS:
                    samples.append({
                        "path": str(vp), "label": label,
                        "source": source, "type": "real" if label == 0 else "t2v",
                    })
    return samples


def load_index_from_csv(csv_path):
    samples = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            samples.append(row)
    return samples


def save_index_to_csv(samples, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "type"])
        writer.writeheader()
        for s in samples:
            writer.writerow(s)
    print(f"  [index] {len(samples)} ornek -> {output_path}")


def preprocess(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data_root) / "index.csv"
    if csv_path.exists():
        print(f"Mevcut index yukleniyor...")
        samples = load_index_from_csv(str(csv_path))
        samples = [s for s in samples if s["source"] not in SKIP_SOURCES]
    else:
        print(f"Index olusturuluyor...")
        samples = build_index_from_dir(args.data_root)
        save_index_to_csv(samples, str(csv_path))

    if args.max_per_class:
        random.seed(42)
        real_s = random.sample([s for s in samples if s["label"] == 0],
                               min(args.max_per_class, sum(1 for s in samples if s["label"] == 0)))
        fake_s = random.sample([s for s in samples if s["label"] == 1],
                               min(args.max_per_class, sum(1 for s in samples if s["label"] == 1)))
        samples = real_s + fake_s
        random.shuffle(samples)

    print(f"Toplam: {len(samples)} | Real: {sum(1 for s in samples if s['label']==0)} | Fake: {sum(1 for s in samples if s['label']==1)}")

    preprocessed_index = []
    tasks = []
    for i, s in enumerate(samples):
        out_name = f"{s['source']}_{Path(s['path']).stem}_{i}.pt"
        out_path = output_dir / out_name
        preprocessed_index.append({"path": str(out_path), "label": s["label"],
                                    "source": s["source"], "type": s["type"]})
        if not out_path.exists():
            tasks.append((s["path"], s["label"], s["source"], s["type"],
                         str(out_path), THIS_DIR, UTILS_DIR))

    print(f"Zaten islenmis: {len(samples) - len(tasks)} | Islenecek: {len(tasks)}")

    if tasks:
        print(f"\nOnisleme basliyor ({args.num_workers} worker)...")
        success = fail = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one, t): t for t in tasks}
            with tqdm(total=len(tasks), desc="Video isleniyor") as pbar:
                for future in as_completed(futures):
                    ok, info = future.result()
                    if ok:
                        success += 1
                    else:
                        fail += 1
                        if fail <= 5:
                            print(f"\n  [HATA] {info}")
                    pbar.update(1)
                    pbar.set_postfix(ok=success, fail=fail)
        print(f"\nTamamlandi: {success} basarili, {fail} hatali")

    valid = [s for s in preprocessed_index if Path(s["path"]).exists()]
    save_index_to_csv(valid, str(output_dir / "preprocessed_index.csv"))
    print(f"Gecerli ornek: {len(valid)}")
    if valid:
        print(f"Disk: ~{Path(valid[0]['path']).stat().st_size * len(valid) / 1e9:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=f"{BASE_DIR}/datasets/genvideo/extracted")
    parser.add_argument("--output_dir", default=f"{BASE_DIR}/datasets/genvideo/preprocessed")
    parser.add_argument("--max_per_class", type=int, default=30000)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args)
