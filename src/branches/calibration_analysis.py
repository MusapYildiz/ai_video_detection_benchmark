import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
calibration_analysis.py - ECE + Reliability Diagram + Risk-Coverage Curve

Adim 4: ECE ve Reliability Diagram
Adim 5: Risk-Coverage Curve (disagreement kullanir)

branch_analysis.py'nin inference ciktisini (JSON) okuyarak calisir,
yeniden model yuklemez — hizli ve bagimsiz.

Eger branch_analysis JSON'u yoksa direkt inference yapar.

Kullanim:
    # Oncelikle branch_analysis.py'yi calistir:
    CUDA_VISIBLE_DEVICES=4 python3 branch_analysis.py --source aegis_raw
    CUDA_VISIBLE_DEVICES=4 python3 branch_analysis.py --source genbuster

    # Sonra bu scripti calistir (model gerektirmez, sadece JSON okur):
    python3 calibration_analysis.py --source aegis_raw
    python3 calibration_analysis.py --source genbuster
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_BASE = Path(f'{BASE_DIR}/checkpoints/phase2/branch_analysis')


# ─── ECE Hesaplama ───────────────────────────

def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error.
    Dusuk ECE: model tahmin ettigi olasiliklara gercekten inanabiliriz.
    Yuksek ECE: model overconfident veya underconfident.
    """
    probs  = np.array(probs)
    labels = np.array(labels)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    bin_stats = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_stats.append(None)
            continue
        bin_conf = probs[mask].mean()
        bin_acc  = labels[mask].mean()
        bin_n    = mask.sum()
        ece     += (bin_n / len(probs)) * abs(bin_acc - bin_conf)
        bin_stats.append({
            'conf': float(bin_conf),
            'acc':  float(bin_acc),
            'n':    int(bin_n),
            'lo':   float(lo),
            'hi':   float(hi),
        })

    return float(ece), bin_stats


# ─── Reliability Diagram ─────────────────────

def plot_reliability_diagram(bin_stats, ece, source, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sol: Reliability diagram
    ax = axes[0]
    valid = [b for b in bin_stats if b is not None]
    confs = [b['conf'] for b in valid]
    accs  = [b['acc']  for b in valid]
    ns    = [b['n']    for b in valid]

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Mukemmel Kalibrasyon')
    ax.bar([b['lo'] for b in valid],
           [b['acc'] for b in valid],
           width=0.1, align='edge', alpha=0.6, color='steelblue', label='Model')
    ax.plot(confs, accs, 'ro-', markersize=5, label='Bin Ortalamasi')

    ax.set_xlabel('Confidence (Tahmin Edilen Olasilik)')
    ax.set_ylabel('Accuracy (Gercek Dogru Oran)')
    ax.set_title(f'Reliability Diagram — {source.upper()}\nECE = {ece:.4f}')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Sag: Bin basi ornek sayisi (kalibrasyon guvenilirligi icin)
    ax2 = axes[1]
    ax2.bar([b['lo'] for b in valid], ns,
            width=0.1, align='edge', alpha=0.7, color='darkorange')
    ax2.set_xlabel('Confidence Bin')
    ax2.set_ylabel('Ornek Sayisi')
    ax2.set_title('Her Confidence Bin\'indeki Ornek Sayisi')

    plt.suptitle(f'Kalibrasyon Analizi — {source.upper()}', fontsize=12)
    plt.tight_layout()
    path = out_dir / f'reliability_diagram_{source}.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Grafik: reliability_diagram_{source}.png")


# ─── Risk-Coverage Curve ─────────────────────

def plot_risk_coverage(records, source, out_dir):
    """
    Risk-Coverage Curve (Adim 5).
    Mantig: En belirsiz (yuksek disagreement) videolari cikartinca
            accuracy artıyor mu?

    Coverage: kac yuzdesi dahil ediliyor (en eminlerden basla)
    Risk:     dahil edilenlerde hata orani (1 - accuracy)
    """
    # Disagreement'a gore sirala (en dusuk = en emin, once dahil et)
    sorted_recs = sorted(records, key=lambda r: r['std'])
    n = len(sorted_recs)

    coverages  = []
    accuracies = []
    thresholds = []

    for cutoff in range(10, n+1, max(1, n//100)):
        subset = sorted_recs[:cutoff]
        acc    = np.mean([r['correct'] for r in subset])
        coverages.append(cutoff / n)
        accuracies.append(acc)
        thresholds.append(np.std([r['pixel'], r['motion'], r['consistency']]
                                  for r in [sorted_recs[cutoff-1]]))

    # Tum veri baseline
    baseline_acc = np.mean([r['correct'] for r in records])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(coverages, accuracies, 'steelblue', linewidth=2, label='Model (Adaptif)')
    ax.axhline(baseline_acc, color='tomato', linestyle='--', linewidth=1.5,
               label=f'Baseline (Tum Veri): {baseline_acc:.3f}')
    ax.fill_between(coverages, baseline_acc, accuracies,
                    where=[a > baseline_acc for a in accuracies],
                    alpha=0.2, color='green', label='Kazanim')
    ax.set_xlabel('Coverage (Dahil Edilen Video Orani)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Risk-Coverage Curve — {source.upper()}\n'
                 f'(Belirsiz videolar cikartilinca accuracy artıyor mu?)')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / f'risk_coverage_{source}.png'
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Grafik: risk_coverage_{source}.png")

    # Coverage noktalari ozeti
    print(f"\nRisk-Coverage Ozeti ({source}):")
    print(f"  Baseline (tum veri):       Accuracy = {baseline_acc:.4f}")
    for cov in [0.50, 0.70, 0.90]:
        idx = min(int(cov * n), len(accuracies)-1)
        print(f"  En emin %{int(cov*100):2d} dahil edilince: Accuracy = {accuracies[idx]:.4f}  "
              f"({'+'if accuracies[idx] > baseline_acc else ''}"
              f"{accuracies[idx]-baseline_acc:+.4f})")

    return {'baseline': float(baseline_acc), 'coverages': coverages, 'accuracies': accuracies}


# ─── Ana ─────────────────────────────────────

def load_records_from_json(source):
    """
    branch_analysis.py'nin JSON ciktisini okur.
    NOT: Bu JSON sadece ozet istatistikleri iceriyor, her video icin
    ham skorlar yok. Bu yuzden inference yapmak gerekecek.
    """
    json_path = OUTPUT_BASE / source / f'branch_analysis_{source}.json'
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def run_inference_for_calibration(source, batch_size=8):
    """
    branch_analysis.py'nin JSON'u sadece ozet iceriyor, ham skorlar yok.
    Kalibrasyon icin her videonun (fusion_prob, label, std) degerleri gerekiyor.
    Bu yuzden hafif bir inference yapiyoruz.
    """
    import sys, os, csv, torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detector_model import VideoForensicsDetector

    CHECKPOINT_PATH = f'{BASE_DIR}/checkpoints/phase2/checkpoint_best.pt'
    AEGIS_RAW_DIR   = f'{BASE_DIR}/datasets/aegis_full/videos/test_data'
    GENBUSTER_DIR_  = f'{BASE_DIR}/datasets/GenBuster-Bench-plusplus/video'
    VAL_INDEX_      = f'{BASE_DIR}/datasets/final_dataset/preprocessed/val/index.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = VideoForensicsDetector(freeze_dino=True).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    print(f"Checkpoint: epoch={state.get('epoch','?')}\n")

    tasks = []
    if source == 'aegis_raw':
        base = Path(AEGIS_RAW_DIR)
        for v in (base/'real'/'youtube').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera'))
        for v in (base/'real'/'dvf').glob('*.mp4'):
            tasks.append((str(v), 0, 'camera'))
        for v in (base/'ai_gen'/'sora').glob('*.mp4'):
            tasks.append((str(v), 1, 'sora'))
        for v in (base/'ai_gen'/'kling').glob('*.mp4'):
            tasks.append((str(v), 1, 'kling'))
    elif source == 'genbuster':
        base = Path(GENBUSTER_DIR_)
        for v in (base/'real').glob('*.mp4'):
            tasks.append((str(v), 0, 'real'))
        for v in (base/'fake').glob('*.mp4'):
            tasks.append((str(v), 1, 'fake'))
    elif source == 'val':
        with open(VAL_INDEX_) as f:
            for r in csv.DictReader(f):
                tasks.append((r['path'], int(r['label']), r['source']))

    print(f"{len(tasks)} video isleniyor...")

    def load_video(path):
        if path.endswith('.pt'):
            return torch.load(path, weights_only=True).float()
        UTILS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
        if UTILS not in sys.path:
            sys.path.insert(0, UTILS)
        from video_io import load_video as lv
        return lv(path, n_frames=16, n_semantic=8, sampling='window',
                  target_dur=4.0, random_start=False, quality_filter=True).frames_all

    records = []
    with torch.no_grad():
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            frames_list, labels, gens = [], [], []
            for path, label, gen in batch:
                try:
                    frames_list.append(load_video(path))
                    labels.append(label); gens.append(gen)
                except Exception:
                    continue
            if not frames_list:
                continue
            frames = torch.stack(frames_list).to(device)
            out = model(frames)
            for j in range(len(frames_list)):
                p = out['pixel_prob'][j].item()
                m = out['motion_prob'][j].item()
                c = out['consistency_prob'][j].item()
                f = out['ai_probability'][j].item()
                scores = np.array([p, m, c])
                records.append({
                    'label': labels[j], 'generator': gens[j],
                    'pixel': p, 'motion': m, 'consistency': c, 'fusion': f,
                    'pred': int(f >= 0.5), 'correct': int(int(f >= 0.5) == labels[j]),
                    'std': float(np.std(scores)), 'var': float(np.var(scores)),
                    'maxmin': float(scores.max() - scores.min()),
                })
            if (i // batch_size) % 25 == 0:
                print(f"  {i+len(batch)}/{len(tasks)} islendi...")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['aegis_raw', 'genbuster', 'val'],
                        default='aegis_raw')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='ECE hesabi icin bin sayisi')
    args = parser.parse_args()

    out_dir = OUTPUT_BASE / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Kaynak: {args.source.upper()}\n")
    print("Inference yapiliyor (kalibrasyon icin ham skorlar gerekiyor)...")
    records = run_inference_for_calibration(args.source)

    probs  = [r['fusion'] for r in records]
    labels = [r['label']  for r in records]

    # ── Adim 4: ECE + Reliability Diagram ──
    print(f"\n{'='*60}")
    print("4. ECE + RELİABİLİTY DİAGRAM")
    print(f"{'='*60}")
    ece, bin_stats = compute_ece(probs, labels, n_bins=args.n_bins)
    print(f"ECE = {ece:.4f}  (0=mukemmel, 0.1+=kotu)")
    valid_bins = [b for b in bin_stats if b is not None]
    for b in valid_bins:
        gap = b['acc'] - b['conf']
        sign = '↑ underconf' if gap > 0.05 else ('↓ overconf' if gap < -0.05 else '✓ iyi')
        print(f"  [{b['lo']:.1f}-{b['hi']:.1f}]  conf={b['conf']:.3f}  "
              f"acc={b['acc']:.3f}  n={b['n']}  {sign}")
    plot_reliability_diagram(bin_stats, ece, args.source, out_dir)

    # ── Adim 5: Risk-Coverage Curve ─────────
    print(f"\n{'='*60}")
    print("5. RİSK-COVERAGE CURVE")
    print(f"{'='*60}")
    rc_results = plot_risk_coverage(records, args.source, out_dir)

    # Kaydet
    summary = {
        'source':    args.source,
        'n':         len(records),
        'ece':       ece,
        'bin_stats': bin_stats,
        'risk_coverage': rc_results,
    }
    out_json = out_dir / f'calibration_{args.source}.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nTum sonuclar kaydedildi: {out_dir}/")


if __name__ == '__main__':
    main()
