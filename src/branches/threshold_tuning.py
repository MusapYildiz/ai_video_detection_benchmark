import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
threshold_tuning.py - Val setinde optimal karar esigini bulur ve
kayitli ham skorlara (scores/*.jsonl) uygulayarak tum test setlerinde
"varsayilan esik (0.5) vs tuned esik" karsilastirmasi yapar.

Hicbir inference calistirmaz - save_scores.py'nin uretmis oldugu
pixel_motion_consistency (ai_probability) skorlarini kullanir.

Mevcut sonuclari (ablation_*.json, full_comparison.*, benchmark_table.*)
DEGISTIRMEZ - ayri dosyalara yazar:
  checkpoints/phase2/threshold_tuning.json          val analizi
  checkpoints/phase2/threshold_comparison.json       once/sonra tum kaynaklar
  checkpoints/phase2/threshold_comparison.md         okunabilir tablo

Kullanim:
    python3 threshold_tuning.py
"""

import json
from pathlib import Path

SCORES_DIR = Path(f'{BASE_DIR}/checkpoints/phase2/scores')
OUT_TUNING = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_tuning.json')
OUT_COMPARISON_JSON = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_comparison.json')
OUT_COMPARISON_MD = Path(f'{BASE_DIR}/checkpoints/phase2/threshold_comparison.md')

SCORE_KEY = 'pixel_motion_consistency'   # = ai_probability, ana kombinasyon
DEFAULT_THRESHOLD = 0.5

# Fake-only kaynaklar: FPR/AUC/precision hesaplanamaz, sadece recall anlamli
FAKEONLY_SOURCES = {'aigvdbench', 'extra_test'}

FULL_SOURCES = {
    'aegis_raw': 'AEGIS Hard (ham video)',
    'genbuster': 'GenBuster-Bench++',
}
FAKEONLY_LABELS = {
    'aigvdbench': 'AIGVDBench',
    'extra_test': 'extra_test (Sora)',
}


def load_scores(source):
    path = SCORES_DIR / f'{source}_scores.jsonl'
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records.append((r['label'], r[SCORE_KEY]))
    return records


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


def find_optimal_thresholds(val_records):
    """Val skorlari uzerinde grid search: F1-max ve Youden's J-max esikleri."""
    thresholds = sorted(set(s for _, s in val_records))
    # kaba + ince grid: hem gorulen skorlar hem de 0.001 adimli grid
    grid = sorted(set(thresholds) | {i / 1000 for i in range(1, 1000)})

    best_f1 = {'threshold': None, 'f1': -1}
    best_j = {'threshold': None, 'j': -2}

    for t in grid:
        m = confusion(val_records, t)
        if m['f1'] > best_f1['f1']:
            best_f1 = {'threshold': t, 'f1': m['f1'], 'metrics': m}
        j = m['recall'] - m['fpr']
        if j > best_j['j']:
            best_j = {'threshold': t, 'j': j, 'metrics': m}

    return best_f1, best_j


def main():
    print("[1/3] Val skorlari yukleniyor...")
    val_records = load_scores('val')
    print(f"      n={len(val_records)}")

    val_auc = compute_auc(val_records)
    default_metrics = confusion(val_records, DEFAULT_THRESHOLD)
    print(f"      Val AUC (esikten bagimsiz): {val_auc:.4f}")
    print(f"      Val @ threshold=0.5000: acc={default_metrics['accuracy']:.4f} "
          f"f1={default_metrics['f1']:.4f} fpr={default_metrics['fpr']:.4f} "
          f"recall={default_metrics['recall']:.4f}")

    print("\n[2/3] Val'de optimal esik araniyor (grid search, F1-max ve Youden J-max)...")
    best_f1, best_j = find_optimal_thresholds(val_records)
    print(f"      F1-optimal:     threshold={best_f1['threshold']:.4f}  "
          f"f1={best_f1['f1']:.4f}  (val'de varsayilana gore delta_f1={best_f1['f1']-default_metrics['f1']:+.4f})")
    print(f"      Youden-J-optimal: threshold={best_j['threshold']:.4f}  "
          f"J={best_j['j']:.4f}  recall={best_j['metrics']['recall']:.4f}  fpr={best_j['metrics']['fpr']:.4f}")

    tuning_result = {
        'score_key': SCORE_KEY,
        'val_n': len(val_records),
        'val_auc': val_auc,
        'default_threshold': DEFAULT_THRESHOLD,
        'default_metrics_on_val': default_metrics,
        'f1_optimal': {'threshold': best_f1['threshold'], 'metrics_on_val': best_f1['metrics']},
        'youden_j_optimal': {'threshold': best_j['threshold'], 'metrics_on_val': best_j['metrics']},
        'chosen_threshold': best_f1['threshold'],
        'chosen_method': 'f1_optimal',
    }
    OUT_TUNING.write_text(json.dumps(tuning_result, indent=2))
    print(f"\n      Kaydedildi: {OUT_TUNING}")

    tuned_threshold = best_f1['threshold']

    print(f"\n[3/3] Tuned esik (threshold={tuned_threshold:.4f}) tum test kaynaklarina uygulaniyor "
          f"(varsayilan 0.5 ile karsilastirma, inference YOK)...")

    comparison = {'tuned_threshold': tuned_threshold, 'default_threshold': DEFAULT_THRESHOLD, 'sources': {}}
    md_lines = ['# Threshold Tuning — Once/Sonra Karsilastirmasi (Bizim Model)\n',
                f"Val setinde F1-maksimize eden esik: **{tuned_threshold:.4f}** "
                f"(varsayilan: {DEFAULT_THRESHOLD})\n",
                "Not: Hicbir inference yeniden calistirilmadi, sadece kayitli ham "
                "olasiliklara (ai_probability) farkli karar esigi uygulandi.\n"]

    md_lines.append("\n## Val (n=%d) — esik secildigi kume\n" % len(val_records))
    md_lines.append("| Esik | Accuracy | Precision | Recall | F1 | FPR |")
    md_lines.append("|---|---|---|---|---|---|")
    for name, t in [('Varsayilan (0.5)', DEFAULT_THRESHOLD), ('Tuned (F1-opt)', tuned_threshold)]:
        m = confusion(val_records, t)
        md_lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {m['f1']:.4f} | {m['fpr']:.4f} |")

    for source, label in FULL_SOURCES.items():
        records = load_scores(source)
        before = confusion(records, DEFAULT_THRESHOLD)
        after = confusion(records, tuned_threshold)
        auc = compute_auc(records)
        comparison['sources'][source] = {'label': label, 'n': len(records), 'auc': auc,
                                          'before': before, 'after': after}
        print(f"\n  {label} (n={len(records)}, AUC={auc:.4f} esikten bagimsiz)")
        print(f"    once (0.5):        acc={before['accuracy']:.4f} f1={before['f1']:.4f} "
              f"recall={before['recall']:.4f} fpr={before['fpr']:.4f}")
        print(f"    sonra ({tuned_threshold:.4f}): acc={after['accuracy']:.4f} f1={after['f1']:.4f} "
              f"recall={after['recall']:.4f} fpr={after['fpr']:.4f}")

        md_lines.append(f"\n## {label} (n={len(records)}, AUC={auc:.4f})\n")
        md_lines.append("| Esik | Accuracy | Precision | Recall | F1 | FPR |")
        md_lines.append("|---|---|---|---|---|---|")
        md_lines.append(f"| Once (0.5) | {before['accuracy']:.4f} | {before['precision']:.4f} | "
                         f"{before['recall']:.4f} | {before['f1']:.4f} | {before['fpr']:.4f} |")
        md_lines.append(f"| Sonra ({tuned_threshold:.4f}) | {after['accuracy']:.4f} | {after['precision']:.4f} | "
                         f"{after['recall']:.4f} | {after['f1']:.4f} | {after['fpr']:.4f} |")
        d_acc = after['accuracy'] - before['accuracy']
        d_f1 = after['f1'] - before['f1']
        d_fpr = after['fpr'] - before['fpr']
        md_lines.append(f"\nDelta: accuracy {d_acc:+.4f}, F1 {d_f1:+.4f}, FPR {d_fpr:+.4f}\n")

    for source, label in FAKEONLY_LABELS.items():
        records = load_scores(source)
        before = recall_only(records, DEFAULT_THRESHOLD)
        after = recall_only(records, tuned_threshold)
        comparison['sources'][source] = {'label': label, 'n': len(records), 'fakeonly': True,
                                          'before': before, 'after': after}
        print(f"\n  {label} (n={len(records)}, sadece FAKE)")
        print(f"    once (0.5):        recall={before['recall']:.4f}")
        print(f"    sonra ({tuned_threshold:.4f}): recall={after['recall']:.4f}")

        md_lines.append(f"\n## {label} (n={len(records)}, sadece FAKE — Recall = tespit orani)\n")
        md_lines.append("| Esik | Recall |")
        md_lines.append("|---|---|")
        md_lines.append(f"| Once (0.5) | {before['recall']:.4f} |")
        md_lines.append(f"| Sonra ({tuned_threshold:.4f}) | {after['recall']:.4f} |")
        md_lines.append(f"\nDelta: recall {after['recall']-before['recall']:+.4f}\n")

    OUT_COMPARISON_JSON.write_text(json.dumps(comparison, indent=2))
    OUT_COMPARISON_MD.write_text('\n'.join(md_lines) + '\n')
    print(f"\nKaydedildi:\n  {OUT_COMPARISON_JSON}\n  {OUT_COMPARISON_MD}")


if __name__ == '__main__':
    main()
