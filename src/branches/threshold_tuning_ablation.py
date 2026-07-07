import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
threshold_tuning_ablation.py - 7 kombinasyonun (pixel, motion, consistency,
pixel+motion, pixel+consistency, motion+consistency, pixel+motion+consistency)
HER BIRI icin ayri ayri: val'de F1-optimal esik bulur, sonra tum test
kaynaklarina (once=0.5 / sonra=tuned) uygular.

Hicbir inference calistirmaz - save_scores.py'nin kayitli ham
olasiliklarini (scores/*.jsonl) kullanir.

Mevcut hicbir dosyayi DEGISTIRMEZ (ne ablation_*.json, ne full_comparison.*,
ne de tekli-kombinasyon threshold_tuning.json/threshold_comparison.*) -
tamamen ayri dosyalara yazar:
  checkpoints/phase2/threshold_tuning_ablation.json      7 kombinasyon x val analizi
  checkpoints/phase2/threshold_comparison_ablation.json  7 kombinasyon x kaynak x once/sonra
  checkpoints/phase2/threshold_comparison_ablation.md    okunabilir tablolar

Kullanim:
    python3 threshold_tuning_ablation.py
"""

import json
from pathlib import Path

SCORES_DIR = Path(f'{BASE_DIR}/checkpoints/phase2/scores')
OUT_TUNING = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_tuning_ablation.json')
OUT_COMPARISON_JSON = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_comparison_ablation.json')
OUT_COMPARISON_MD = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_comparison_ablation.md')

DEFAULT_THRESHOLD = 0.5

COMBINATIONS = [
    ('pixel', 'Pixel'),
    ('motion', 'Motion'),
    ('consistency', 'Consistency'),
    ('pixel_motion', 'Pixel+Motion'),
    ('pixel_consistency', 'Pixel+Consistency'),
    ('motion_consistency', 'Motion+Consistency'),
    ('pixel_motion_consistency', 'Pixel+Motion+Consistency (ana)'),
]

FULL_SOURCES = {
    'aegis_raw': 'AEGIS Hard (ham video)',
    'genbuster': 'GenBuster-Bench++',
}
FAKEONLY_LABELS = {
    'aigvdbench': 'AIGVDBench',
    'extra_test': 'extra_test (Sora)',
}

_cache = {}


def load_scores(source, key):
    """(label, score) listesi. Kaynak basina bir kez dosyayi okuyup cache'ler."""
    if source not in _cache:
        path = SCORES_DIR / f'{source}_scores.jsonl'
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        _cache[source] = rows
    return [(r['label'], r[key]) for r in _cache[source]]


def confusion(records, threshold):
    tp = tn = fp = fn = 0
    for label, score in records:
        pred = 1 if score >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 1 and pred == 0:
            fn += 1
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {'n': n, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fpr}


def recall_only(records, threshold):
    detected = sum(1 for label, score in records if score >= threshold)
    n = len(records)
    return {'n': n, 'detected': detected, 'missed': n - detected,
            'recall': detected / n if n else 0.0}


def compute_auc(records):
    pos = [s for l, s in records if l == 1]
    neg = [s for l, s in records if l == 0]
    if not pos or not neg:
        return None
    count = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (len(pos) * len(neg))


def find_f1_optimal(val_records):
    grid = sorted(set(s for _, s in val_records) | {i / 1000 for i in range(1, 1000)})
    best = {'threshold': None, 'f1': -1}
    for t in grid:
        m = confusion(val_records, t)
        if m['f1'] > best['f1']:
            best = {'threshold': t, 'f1': m['f1'], 'metrics': m}
    return best


def main():
    tuning_out = {}
    comparison_out = {}
    md_lines = ['# 7 Kombinasyon — Threshold Tuning Once/Sonra (Bizim Model)\n',
                "Her kombinasyon icin ayri ayri: val'de F1-maksimize eden esik bulunur, "
                "sonra bu esik o kombinasyonun tum test kaynaklarindaki ham skorlarina "
                "uygulanir. Hicbir inference yeniden calistirilmadi.\n"]

    print(f"[1/2] Val skorlari 7 kombinasyon icin ayri ayri analiz ediliyor...")
    for key, label in COMBINATIONS:
        val_records = load_scores('val', key)
        val_auc = compute_auc(val_records)
        default_val = confusion(val_records, DEFAULT_THRESHOLD)
        best = find_f1_optimal(val_records)
        tuned_t = best['threshold']

        tuning_out[key] = {
            'label': label, 'val_n': len(val_records), 'val_auc': val_auc,
            'default_threshold': DEFAULT_THRESHOLD, 'default_metrics_on_val': default_val,
            'tuned_threshold': tuned_t, 'tuned_metrics_on_val': best['metrics'],
        }
        print(f"  {label:<32} val_auc={val_auc:.4f}  tuned_threshold={tuned_t:.4f}  "
              f"(val F1 {default_val['f1']:.4f} -> {best['metrics']['f1']:.4f})")

    OUT_TUNING.write_text(json.dumps(tuning_out, indent=2))
    print(f"\n  Kaydedildi: {OUT_TUNING}")

    print(f"\n[2/2] Her kombinasyonun kendi tuned esigi tum kaynaklara uygulaniyor...")

    def dataset_table(source, label, fakeonly):
        rows_before, rows_after = [], []
        for key, clabel in COMBINATIONS:
            records = load_scores(source, key)
            tuned_t = tuning_out[key]['tuned_threshold']
            if fakeonly:
                before = recall_only(records, DEFAULT_THRESHOLD)
                after = recall_only(records, tuned_t)
                auc = None
            else:
                before = confusion(records, DEFAULT_THRESHOLD)
                after = confusion(records, tuned_t)
                auc = compute_auc(records)
            rows_before.append((clabel, before))
            rows_after.append((clabel, after, tuned_t, auc))
        return rows_before, rows_after

    all_sources = list(FULL_SOURCES.items()) + list(FAKEONLY_LABELS.items())
    for source, label in all_sources:
        fakeonly = source in FAKEONLY_LABELS
        rows_before, rows_after = dataset_table(source, label, fakeonly)
        n = rows_after[0][1]['n']
        comparison_out[source] = {
            'label': label, 'n': n, 'fakeonly': fakeonly,
            'combinations': {
                key: {
                    'label': clabel,
                    'tuned_threshold': rows_after[i][2],
                    'auc': rows_after[i][3],
                    'before': rows_before[i][1],
                    'after': rows_after[i][1],
                } for i, (key, clabel) in enumerate(COMBINATIONS)
            }
        }

        print(f"\n  === {label} (n={n}{', sadece FAKE' if fakeonly else ''}) ===")
        md_lines.append(f"\n## {label} (n={n}{', sadece FAKE video — Recall = tespit orani' if fakeonly else ''})\n")

        if fakeonly:
            print(f"  {'Kombinasyon':<32} {'Esik':<8} {'Recall once':<13} {'Recall sonra':<13} {'Delta':<8}")
            md_lines.append("| Kombinasyon | Tuned Esik | Recall (once, 0.5) | Recall (sonra) | Delta |")
            md_lines.append("|---|---|---|---|---|")
            for i, (key, clabel) in enumerate(COMBINATIONS):
                before = rows_before[i][1]
                _, after, tuned_t, _ = rows_after[i]
                delta = after['recall'] - before['recall']
                print(f"  {clabel:<32} {tuned_t:<8.4f} {before['recall']:<13.4f} {after['recall']:<13.4f} {delta:+.4f}")
                md_lines.append(f"| {clabel} | {tuned_t:.4f} | {before['recall']:.4f} | {after['recall']:.4f} | {delta:+.4f} |")
        else:
            print(f"  {'Kombinasyon':<32} {'AUC':<8} {'Esik':<8} {'Acc(0.5->tuned)':<20} {'F1(0.5->tuned)':<20} {'FPR(0.5->tuned)':<20}")
            md_lines.append("| Kombinasyon | AUC | Tuned Esik | Accuracy once | Accuracy sonra | F1 once | F1 sonra | FPR once | FPR sonra |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for i, (key, clabel) in enumerate(COMBINATIONS):
                before = rows_before[i][1]
                _, after, tuned_t, auc = rows_after[i]
                print(f"  {clabel:<32} {auc:<8.4f} {tuned_t:<8.4f} "
                      f"{before['accuracy']:.4f}->{after['accuracy']:.4f}      "
                      f"{before['f1']:.4f}->{after['f1']:.4f}      "
                      f"{before['fpr']:.4f}->{after['fpr']:.4f}")
                md_lines.append(f"| {clabel} | {auc:.4f} | {tuned_t:.4f} | {before['accuracy']:.4f} | "
                                 f"{after['accuracy']:.4f} | {before['f1']:.4f} | {after['f1']:.4f} | "
                                 f"{before['fpr']:.4f} | {after['fpr']:.4f} |")

    OUT_COMPARISON_JSON.write_text(json.dumps(comparison_out, indent=2))
    OUT_COMPARISON_MD.write_text('\n'.join(md_lines) + '\n')
    print(f"\nKaydedildi:\n  {OUT_COMPARISON_JSON}\n  {OUT_COMPARISON_MD}")


if __name__ == '__main__':
    main()
