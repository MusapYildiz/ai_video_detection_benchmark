import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
branch_analysis.py - Branch Disagreement ve Correlation Analizi

Mevcut checkpoint'ten yeniden inference yaparak her video icin
branch skorlarini, disagreement metriklerini ve correlation'i hesaplar.

Analizler:
  1. Branch Disagreement: Yanlis tahminlerde std/variance daha yuksek mi?
  2. Branch Correlation: Uc branch ayni seyleri mi ogreniyor?
  3. Confidence Histogram: Real/fake dagilimi ne kadar ayrisik?
  4. Per-Source Analysis: Hangi generator zor, hangisi kolay?

Kullanim:
    CUDA_VISIBLE_DEVICES=4 python3 branch_analysis.py --source aegis_raw
    CUDA_VISIBLE_DEVICES=4 python3 branch_analysis.py --source genbuster
    CUDA_VISIBLE_DEVICES=4 python3 branch_analysis.py --source val
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ekransiz ortam icin
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector_model import VideoForensicsDetector

# ─── Sabitler ───────────────────────────────
CHECKPOINT        = f'{BASE_DIR}/checkpoints/phase2/checkpoint_best.pt'
AEGIS_RAW_DIR     = f'{BASE_DIR}/datasets/aegis_full/videos/test_data'
GENBUSTER_DIR     = f'{BASE_DIR}/datasets/GenBuster-Bench-plusplus/video'
VAL_INDEX         = f'{BASE_DIR}/datasets/final_dataset/preprocessed/val/index.csv'
OUTPUT_DIR        = Path(f'{BASE_DIR}/checkpoints/phase2/branch_analysis')

# ─── Video yukleme ──────────────────────────

def _load_raw_video(path, n_frames=16):
    UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if UTILS_DIR not in sys.path:
        sys.path.insert(0, UTILS_DIR)
    from video_io import load_video
    bundle = load_video(path, n_frames=n_frames, n_semantic=8,
                        sampling='window', target_dur=4.0,
                        random_start=False, quality_filter=True)
    return bundle.frames_all


# ─── Inference (tum branch skorlarini topla) ─

@torch.no_grad()
def collect_scores(model, device, source, batch_size=8):
    """
    Tum videolardan inference yaparak her video icin:
      - pixel_prob, motion_prob, consistency_prob (tekil branch'ler)
      - ai_probability (ana fusion)
      - label (0=real, 1=fake)
      - source/generator bilgisi
    dondurur.
    """
    tasks = []  # (path, label, generator)

    if source == 'aegis_raw':
        base = Path(AEGIS_RAW_DIR)
        for v in (base / 'real' / 'youtube').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera_youtube'))
        for v in (base / 'real' / 'dvf').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera_dvf'))
        for v in (base / 'ai_gen' / 'sora').glob('*.mp4'):
            tasks.append((str(v), 1, 'sora'))
        for v in (base / 'ai_gen' / 'kling').glob('*.mp4'):
            tasks.append((str(v), 1, 'kling'))

    elif source == 'genbuster':
        base = Path(GENBUSTER_DIR)
        for v in (base / 'real').glob('*.mp4'):
            tasks.append((str(v), 0, 'real'))
        for v in (base / 'fake').glob('*.mp4'):
            tasks.append((str(v), 1, 'fake'))

    elif source == 'val':
        with open(VAL_INDEX) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            tasks.append((r['path'], int(r['label']), r['source']))

    print(f"Toplam {len(tasks)} video isleniyor...\n")

    records = []
    n_skipped = 0

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        frames_list, labels, gens, paths = [], [], [], []

        for path, label, gen in batch:
            try:
                if path.endswith('.pt'):
                    t = torch.load(path, weights_only=True).float()
                else:
                    t = _load_raw_video(path)
                frames_list.append(t)
                labels.append(label)
                gens.append(gen)
                paths.append(path)
            except Exception:
                n_skipped += 1
                continue

        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        out = model(frames)

        for j in range(len(frames_list)):
            pixel  = out['pixel_prob'][j].item()
            motion = out['motion_prob'][j].item()
            consist= out['consistency_prob'][j].item()
            fusion = out['ai_probability'][j].item()

            scores = np.array([pixel, motion, consist])
            std     = float(np.std(scores))
            var     = float(np.var(scores))
            maxmin  = float(scores.max() - scores.min())
            entropy = float(-np.sum(
                np.clip(scores, 1e-9, 1) * np.log(np.clip(scores, 1e-9, 1)) +
                np.clip(1-scores, 1e-9, 1) * np.log(np.clip(1-scores, 1e-9, 1))
            ) / 3)

            pred   = int(fusion >= 0.5)
            correct= int(pred == labels[j])

            records.append({
                'path':        paths[j],
                'label':       labels[j],
                'generator':   gens[j],
                'pixel':       pixel,
                'motion':      motion,
                'consistency': consist,
                'fusion':      fusion,
                'pred':        pred,
                'correct':     correct,
                'std':         std,
                'var':         var,
                'maxmin':      maxmin,
                'entropy':     entropy,
            })

        if (i // batch_size) % 25 == 0:
            print(f"  {i+len(batch)}/{len(tasks)} islendi...")

    print(f"Tamamlandi: {len(records)} video | Atlanan: {n_skipped}\n")
    return records


# ─── Analiz 1: Branch Disagreement ──────────

def analyze_disagreement(records, out_dir, source):
    correct   = [r for r in records if r['correct'] == 1]
    incorrect = [r for r in records if r['correct'] == 0]

    print(f"{'='*60}")
    print("1. BRANCH DISAGREEMENT ANALİZİ")
    print(f"{'='*60}")
    print(f"Dogru tahmin:   {len(correct)}")
    print(f"Yanlis tahmin:  {len(incorrect)}\n")

    metrics = ['std', 'var', 'maxmin', 'entropy']
    results = {}
    for m in metrics:
        c_vals = [r[m] for r in correct]
        w_vals = [r[m] for r in incorrect]
        print(f"{m:<10}  Dogru: {np.mean(c_vals):.4f} ± {np.std(c_vals):.4f} | "
              f"Yanlis: {np.mean(w_vals):.4f} ± {np.std(w_vals):.4f}")
        results[m] = {
            'correct_mean': float(np.mean(c_vals)),
            'correct_std':  float(np.std(c_vals)),
            'wrong_mean':   float(np.mean(w_vals)),
            'wrong_std':    float(np.std(w_vals)),
        }

    # std dagilim grafigi
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, subset, title, color in [
        (axes[0], correct,   'Dogru Tahminler', 'steelblue'),
        (axes[1], incorrect, 'Yanlis Tahminler', 'tomato'),
    ]:
        ax.hist([r['std'] for r in subset], bins=30, color=color, alpha=0.7)
        ax.set_title(f"{title} (n={len(subset)})")
        ax.set_xlabel('Branch STD (disagreement)')
        ax.set_ylabel('Video Sayisi')
        ax.axvline(np.mean([r['std'] for r in subset]),
                   color='black', linestyle='--', label='Ortalama')
        ax.legend()
    plt.suptitle(f'Branch Disagreement Dagilimi — {source.upper()}')
    plt.tight_layout()
    fig.savefig(out_dir / f'disagreement_histogram_{source}.png', dpi=150)
    plt.close()
    print(f"\nGrafik: disagreement_histogram_{source}.png")
    return results


# ─── Analiz 2: Branch Correlation ───────────

def analyze_correlation(records, out_dir, source):
    pixel  = np.array([r['pixel']       for r in records])
    motion = np.array([r['motion']      for r in records])
    consist= np.array([r['consistency'] for r in records])

    print(f"\n{'='*60}")
    print("2. BRANCH CORRELATION ANALİZİ")
    print(f"{'='*60}")

    corr_pm = float(np.corrcoef(pixel,  motion)[0, 1])
    corr_pc = float(np.corrcoef(pixel,  consist)[0, 1])
    corr_mc = float(np.corrcoef(motion, consist)[0, 1])

    print(f"Pixel  ↔ Motion:      r = {corr_pm:.4f}")
    print(f"Pixel  ↔ Consistency: r = {corr_pc:.4f}")
    print(f"Motion ↔ Consistency: r = {corr_mc:.4f}")
    print()
    for name, val in [('Pixel↔Motion', corr_pm),
                       ('Pixel↔Consistency', corr_pc),
                       ('Motion↔Consistency', corr_mc)]:
        if abs(val) > 0.8:
            print(f"  [UYARI] {name}: cok yuksek korelasyon ({val:.3f}) — branch'ler ayni seyi ogreniyor olabilir")
        elif abs(val) < 0.3:
            print(f"  [IYI]   {name}: dusuk korelasyon ({val:.3f}) — tamamlayici bilgi tasiyorlar")
        else:
            print(f"  [ORTA]  {name}: orta korelasyon ({val:.3f})")

    # Scatter matrix
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels_arr = np.array([r['label'] for r in records])
    colors = ['steelblue' if l == 0 else 'tomato' for l in labels_arr]

    for ax, (xa, ya, xname, yname) in zip(axes, [
        (pixel, motion,  'Pixel',  'Motion'),
        (pixel, consist, 'Pixel',  'Consistency'),
        (motion, consist,'Motion', 'Consistency'),
    ]):
        ax.scatter(xa, ya, c=colors, alpha=0.3, s=8)
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        r = float(np.corrcoef(xa, ya)[0, 1])
        ax.set_title(f"r = {r:.3f}")

    # Efsane
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Real'),
                       Patch(facecolor='tomato',    label='Fake')]
    axes[2].legend(handles=legend_elements, loc='lower right')
    plt.suptitle(f'Branch Korelasyon — {source.upper()}')
    plt.tight_layout()
    fig.savefig(out_dir / f'branch_correlation_{source}.png', dpi=150)
    plt.close()
    print(f"Grafik: branch_correlation_{source}.png")

    return {'pixel_motion': corr_pm, 'pixel_consistency': corr_pc,
            'motion_consistency': corr_mc}


# ─── Analiz 3: Confidence Histogram ─────────

def analyze_confidence_histogram(records, out_dir, source):
    real_probs = [r['fusion'] for r in records if r['label'] == 0]
    fake_probs = [r['fusion'] for r in records if r['label'] == 1]

    print(f"\n{'='*60}")
    print("3. CONFIDENCE HISTOGRAM")
    print(f"{'='*60}")
    print(f"Real (n={len(real_probs)}): ortalama={np.mean(real_probs):.3f}, "
          f"std={np.std(real_probs):.3f}")
    print(f"Fake (n={len(fake_probs)}): ortalama={np.mean(fake_probs):.3f}, "
          f"std={np.std(fake_probs):.3f}")

    if len(real_probs) > 0 and len(fake_probs) > 0:
        all_labels = [0]*len(real_probs) + [1]*len(fake_probs)
        all_probs  = real_probs + fake_probs
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC: {auc:.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, 1, 41)
    if real_probs:
        ax.hist(real_probs, bins=bins, alpha=0.6, color='steelblue',
                label=f'Real (n={len(real_probs)})', density=True)
    if fake_probs:
        ax.hist(fake_probs, bins=bins, alpha=0.6, color='tomato',
                label=f'Fake (n={len(fake_probs)})', density=True)
    ax.axvline(0.5, color='black', linestyle='--', label='Esik (0.5)')
    ax.set_xlabel('AI Probability (Fusion)')
    ax.set_ylabel('Yogunluk')
    ax.set_title(f'Confidence Histogram — {source.upper()}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / f'confidence_histogram_{source}.png', dpi=150)
    plt.close()
    print(f"Grafik: confidence_histogram_{source}.png")


# ─── Analiz 4: Per-Source Analysis ──────────

def analyze_per_source(records, out_dir, source):
    from collections import defaultdict

    print(f"\n{'='*60}")
    print("4. PER-SOURCE / GENERATOR ANALİZİ")
    print(f"{'='*60}")

    groups = defaultdict(list)
    for r in records:
        groups[r['generator']].append(r)

    print(f"{'Generator':<20} {'N':<6} {'Acc':<8} {'Recall':<8} "
          f"{'FPR':<8} {'Ort.Fusion':<12} {'Ort.STD':<10}")
    print('-' * 75)

    per_source_results = {}
    for gen in sorted(groups.keys()):
        recs = groups[gen]
        n = len(recs)
        acc = np.mean([r['correct'] for r in recs])
        fusions = [r['fusion'] for r in recs]
        stds    = [r['std']    for r in recs]
        labels  = [r['label']  for r in recs]
        preds   = [r['pred']   for r in recs]

        # recall (sadece fake icin anlamli)
        fakes = [r for r in recs if r['label'] == 1]
        recall = np.mean([r['correct'] for r in fakes]) if fakes else float('nan')
        # fpr (sadece real icin anlamli)
        reals = [r for r in recs if r['label'] == 0]
        fpr = np.mean([1 - r['correct'] for r in reals]) if reals else float('nan')

        print(f"{gen:<20} {n:<6} {acc:<8.4f} {recall:<8.4f} "
              f"{fpr:<8.4f} {np.mean(fusions):<12.4f} {np.mean(stds):<10.4f}")
        per_source_results[gen] = {
            'n': n, 'accuracy': float(acc), 'recall': float(recall),
            'fpr': float(fpr), 'mean_fusion': float(np.mean(fusions)),
            'mean_std': float(np.mean(stds)),
        }

    return per_source_results


# ─── Ana ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['aegis_raw', 'genbuster', 'val'],
                        default='aegis_raw')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Model yukle
    model = VideoForensicsDetector(freeze_dino=True).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    epoch = state.get('epoch', '?')
    print(f"Checkpoint: epoch={epoch}\n")

    # Cikti dizini
    out_dir = OUTPUT_DIR / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    records = collect_scores(model, device, args.source, args.batch_size)

    # Analizler
    disagree_results  = analyze_disagreement(records, out_dir, args.source)
    corr_results      = analyze_correlation(records, out_dir, args.source)
    analyze_confidence_histogram(records, out_dir, args.source)
    per_source        = analyze_per_source(records, out_dir, args.source)

    # Kaydet
    summary = {
        'source':            args.source,
        'checkpoint':        args.checkpoint,
        'n_total':           len(records),
        'n_real':            sum(1 for r in records if r['label'] == 0),
        'n_fake':            sum(1 for r in records if r['label'] == 1),
        'overall_accuracy':  float(np.mean([r['correct'] for r in records])),
        'disagreement':      disagree_results,
        'correlation':       corr_results,
        'per_source':        per_source,
    }
    out_json = out_dir / f'branch_analysis_{args.source}.json'
    import json as _json
    with open(out_json, 'w') as f:
        _json.dump(summary, f, indent=2)
    print(f"\nTum sonuclar kaydedildi: {out_dir}/")


if __name__ == '__main__':
    main()
