import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
save_scores.py - Tum test kaynaklarinda, 7 kombinasyon icin ham skorlari kaydet.

Bir kez calistir, bir daha inference yapma.
Sonraki tum analizler (threshold tuning, ECE, risk-coverage, disagreement,
ablation karsilastirma) bu dosyadan okur.

Kaydedilen bilgi (her video icin):
  - path, label, source/generator
  - 7 kombinasyonun ai_probability degerleri
  - pixel/motion/consistency tekil olasılıkları
  - branch std/var (disagreement)

Cikti: /checkpoints/phase2/scores/{source}_scores.jsonl
       (her satir bir video, JSON Lines formati — buyuk dosyalar icin ideal)

Kullanim:
    CUDA_VISIBLE_DEVICES=4 python3 save_scores.py --source all
    CUDA_VISIBLE_DEVICES=4 python3 save_scores.py --source aegis_raw
    CUDA_VISIBLE_DEVICES=4 python3 save_scores.py --source genbuster
"""

import os, sys, csv, json, argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_model import VideoForensicsDetector

CHECKPOINT    = f'{BASE_DIR}/checkpoints/phase2/checkpoint_best.pt'
VAL_INDEX     = f'{BASE_DIR}/datasets/final_dataset/preprocessed/val/index.csv'
AEGIS_RAW_DIR = f'{BASE_DIR}/datasets/aegis_full/videos/test_data'
GENBUSTER_DIR = f'{BASE_DIR}/datasets/GenBuster-Bench-plusplus/video'
AIGVDBENCH_DIR= f'{BASE_DIR}/datasets/aigvdbench/extracted/fake'
EXTRA_TEST_CSV= f'{BASE_DIR}/datasets/final_dataset/extra_test.csv'
OUT_DIR       = Path(f'{BASE_DIR}/checkpoints/phase2/scores')

# 7 kombinasyon — model ciktisindaki key -> kayit adi eslemesi
COMBINATIONS = [
    ('pixel',                      'pixel_prob'),
    ('motion',                     'motion_prob'),
    ('consistency',                'consistency_prob'),
    ('pixel_motion',               'pixel_motion_prob'),
    ('pixel_consistency',          'pixel_consistency_prob'),
    ('motion_consistency',         'motion_consistency_prob'),
    ('pixel_motion_consistency',   'ai_probability'),
]


def load_video_tensor(path):
    if path.endswith('.pt'):
        return torch.load(path, weights_only=True).float()
    UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if UTILS not in sys.path:
        sys.path.insert(0, UTILS)
    from video_io import load_video
    return load_video(path, n_frames=16, n_semantic=8, sampling='window',
                      target_dur=4.0, random_start=False,
                      quality_filter=True).frames_all


def build_tasks(source):
    """(path, label, generator) listesi dondurur."""
    tasks = []
    if source == 'val':
        with open(VAL_INDEX) as f:
            for r in csv.DictReader(f):
                tasks.append((r['path'], int(r['label']), r['source']))

    elif source == 'aegis_raw':
        base = Path(AEGIS_RAW_DIR)
        for v in (base/'real'/'youtube').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera_youtube'))
        for v in (base/'real'/'dvf').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera_dvf'))
        for v in (base/'ai_gen'/'sora').glob('*.mp4'):
            tasks.append((str(v), 1, 'sora'))
        for v in (base/'ai_gen'/'kling').glob('*.mp4'):
            tasks.append((str(v), 1, 'kling'))

    elif source == 'genbuster':
        base = Path(GENBUSTER_DIR)
        for v in (base/'real').glob('*.mp4'):
            tasks.append((str(v), 0, 'real'))
        for v in (base/'fake').glob('*.mp4'):
            tasks.append((str(v), 1, 'fake'))

    elif source == 'aigvdbench':
        base = Path(AIGVDBENCH_DIR)
        for model_dir in sorted(base.iterdir()):
            if model_dir.is_dir():
                for v in model_dir.glob('*.mp4'):
                    tasks.append((str(v), 1, model_dir.name))

    elif source == 'extra_test':
        with open(EXTRA_TEST_CSV) as f:
            for r in csv.DictReader(f):
                tasks.append((r['path'], int(r['label']), r.get('source', 'sora')))

    return tasks


@torch.no_grad()
def run_and_save(model, device, source, batch_size=8):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{source}_scores.jsonl'

    tasks = build_tasks(source)
    print(f"\n[{source}] {len(tasks)} video isleniyor → {out_path}")

    n_written = 0
    n_skipped = 0

    with open(out_path, 'w') as fout:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            frames_list, paths, labels, gens = [], [], [], []

            for path, label, gen in batch:
                try:
                    frames_list.append(load_video_tensor(path))
                    paths.append(path)
                    labels.append(label)
                    gens.append(gen)
                except Exception:
                    n_skipped += 1
                    continue

            if not frames_list:
                continue

            frames = torch.stack(frames_list).to(device)
            out = model(frames)

            for j in range(len(frames_list)):
                scores_7 = {name: float(out[key][j].item())
                            for name, key in COMBINATIONS}

                p = scores_7['pixel']
                m = scores_7['motion']
                c = scores_7['consistency']
                branch_arr = np.array([p, m, c])

                record = {
                    'path':      paths[j],
                    'label':     labels[j],
                    'generator': gens[j],
                    **scores_7,
                    'branch_std':    float(np.std(branch_arr)),
                    'branch_var':    float(np.var(branch_arr)),
                    'branch_maxmin': float(branch_arr.max() - branch_arr.min()),
                }
                fout.write(json.dumps(record) + '\n')
                n_written += 1

            if (i // batch_size) % 25 == 0:
                print(f"  {i+len(batch)}/{len(tasks)} islendi...")

    print(f"  Kaydedildi: {n_written} video | Atlanan: {n_skipped}")
    print(f"  Dosya: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)\n")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        choices=['val', 'aegis_raw', 'genbuster',
                                 'aigvdbench', 'extra_test', 'all'],
                        default='all')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint',  type=str, default=CHECKPOINT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = VideoForensicsDetector(freeze_dino=True).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    print(f"Checkpoint: epoch={state.get('epoch','?')}")
    print(f"Cikti dizini: {OUT_DIR}\n")

    sources = (['val', 'aegis_raw', 'genbuster', 'aigvdbench', 'extra_test']
               if args.source == 'all' else [args.source])

    for src in sources:
        run_and_save(model, device, src, args.batch_size)

    print("=" * 60)
    print("TUMU TAMAMLANDI")
    print(f"Dosyalar: {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob('*.jsonl')):
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines} video, {f.stat().st_size/1e6:.1f} MB")


if __name__ == '__main__':
    main()
