import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
build_full_comparison.py - Bizim model + diger 6 modelin TUM test
setlerindeki (AEGIS, GenBuster, AIGVDBench, extra_test) karsilastirmasi.

Threshold tuning YAPILMAMIS durumda (varsayilan 0.5 esik / modellerin
kendi ic karar mekanizmasi). ablation_eval.py ve compute_metrics.py
ciktilarindan derlenir, hicbir yeniden inference gerektirmez.

Kaynaklar:
  Bizim model : checkpoints/phase2/ablation_{aegis_raw,genbuster,aigvdbench,extra_test}.json
  Diger 5     : other_models/{comparison_report,report_genbuster,report_aigvdbench,report_extratest}.csv
  D3          : other_models/D3/results/*.txt (AP skoru, ayri metodoloji)

Kullanim:
    python3 build_full_comparison.py
"""

import csv
import json
from pathlib import Path

CKPT_DIR = Path(f'{BASE_DIR}/checkpoints/phase2')
OTHER_DIR = Path(f'{BASE_DIR}/ai_video_detector/other_models')
D3_RESULTS_DIR = OTHER_DIR / 'D3' / 'results'

OUTPUT_MD = CKPT_DIR / 'full_comparison.md'
OUTPUT_CSV = CKPT_DIR / 'full_comparison.csv'

OUR_MODEL_NAME = 'Bizim Model (ham video)'

FULL_DATASETS = [
    # (label, our_json, other_csv, n_total)
    ('AEGIS Hard (ham video)', CKPT_DIR / 'ablation_aegis_raw.json', OTHER_DIR / 'comparison_report.csv', 436),
    ('GenBuster-Bench++', CKPT_DIR / 'ablation_genbuster.json', OTHER_DIR / 'report_genbuster.csv', 2000),
]

FAKEONLY_DATASETS = [
    ('AIGVDBench', CKPT_DIR / 'ablation_aigvdbench.json', OTHER_DIR / 'report_aigvdbench.csv', 250),
    ('extra_test (Sora)', CKPT_DIR / 'ablation_extra_test.json', OTHER_DIR / 'report_extratest.csv', 51),
]


def load_our_full(path):
    with open(path) as f:
        d = json.load(f)
    ana = d['metrics']['pixel+motion+consistency (ana)']
    perf = d['performance']
    info = d['model_info']
    return {
        'model': OUR_MODEL_NAME,
        'accuracy': ana['accuracy'], 'precision': ana['precision'],
        'recall': ana['recall'], 'f1': ana['f1'], 'auc': ana['auc'],
        'avg_time_s': perf['avg_time_per_video_s'],
        'params_b': info['n_params_total'] / 1e9,
        'disk_gb': info['checkpoint_size_mb'] / 1024,
        'peak_gpu_gb': perf['peak_gpu_memory_mb'] / 1024 if perf.get('peak_gpu_memory_mb') else None,
    }


def load_other_full(path):
    rows = []
    if not path.exists():
        print(f"[UYARI] {path} bulunamadi.")
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['category'] != 'TOPLU (ALL)':
                continue
            auc_val = r.get('auc', '')
            rows.append({
                'model': r['model'],
                'accuracy': float(r['accuracy']), 'precision': float(r['precision']),
                'recall': float(r['recall']), 'f1': float(r['f1']),
                'auc': float(auc_val) if auc_val not in (None, '', 'N/A') else None,
                'avg_time_s': float(r['mean_seconds']),
                'params_b': float(r['param_count_billions']),
                'disk_gb': float(r['disk_size_gb']),
                'peak_gpu_gb': float(r['peak_gpu_memory_gb']),
            })
    return rows


def load_our_fakeonly(path):
    with open(path) as f:
        d = json.load(f)
    ana = d['metrics']['pixel+motion+consistency (ana)']
    perf = d['performance']
    return {
        'model': OUR_MODEL_NAME, 'n': ana['n'],
        'recall': ana['recall'], 'avg_time_s': perf['avg_time_per_video_s'],
    }


def load_other_fakeonly(path):
    rows = []
    if not path.exists():
        print(f"[UYARI] {path} bulunamadi.")
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['category'] != 'TOPLU (ALL)':
                continue
            rows.append({
                'model': r['model'], 'n': int(r['n']),
                'recall': float(r['recall']),
                'avg_time_s': float(r['mean_seconds']) if r['mean_seconds'] else None,
            })
    return rows


def load_d3_results():
    """D3/results/*.txt -> {generator: ap_score}."""
    results = {}
    if not D3_RESULTS_DIR.exists():
        return results
    for txt_file in sorted(D3_RESULTS_DIR.glob('*.txt')):
        content = txt_file.read_text()
        fake_csv = ap_score = None
        for line in content.splitlines():
            if line.startswith('Fake CSV:'):
                fake_csv = line.split(':', 1)[1].strip()
            elif line.startswith('AP Score:'):
                ap_score = float(line.split(':', 1)[1].strip())
        if fake_csv and ap_score is not None:
            results[Path(fake_csv).stem] = ap_score
    return results


def fmt(v, spec='.4f'):
    return format(v, spec) if v is not None else 'N/A'


def main():
    md_lines = ['# AEGIS — Genel Model Karsilastirmasi (Tum Test Setleri)\n']
    md_lines.append('Not: Threshold tuning YAPILMAMIS durumdaki (varsayilan karar esigi) sonuclardir.\n')
    csv_rows = [['dataset', 'model', 'n', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                 'avg_time_s', 'params_b', 'disk_gb', 'peak_gpu_gb']]

    d3 = load_d3_results()

    # --- Full (real+fake) datasets ---
    for label, our_json, other_csv, n_total in FULL_DATASETS:
        rows = []
        if our_json.exists():
            rows.append(load_our_full(our_json))
        rows += load_other_full(other_csv)
        rows_sorted = sorted(rows, key=lambda r: (r['auc'] is None, -(r['auc'] or 0)))

        print(f"\n{'='*115}")
        print(f"{label} (n={n_total})")
        print(f"{'='*115}")
        header = f"{'Model':<26} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'AUC':<8} {'Sure(s)':<9} {'Param(B)':<9} {'Disk(GB)':<9} {'GPU(GB)':<8}"
        print(header)
        print('-' * len(header))
        for r in rows_sorted:
            print(f"{r['model']:<26} {r['accuracy']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} "
                  f"{r['f1']:<8.4f} {fmt(r['auc']):<8} {r['avg_time_s']:<9.3f} {r['params_b']:<9.3f} "
                  f"{r['disk_gb']:<9.2f} {r['peak_gpu_gb']:<8.2f}")

        md_lines.append(f"\n## {label} (n={n_total})\n")
        md_lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC | Sure(s) | Param(B) | Disk(GB) | GPU(GB) |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows_sorted:
            md_lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                             f"{r['recall']:.4f} | {r['f1']:.4f} | {fmt(r['auc'])} | {r['avg_time_s']:.3f} | "
                             f"{r['params_b']:.3f} | {r['disk_gb']:.2f} | {r['peak_gpu_gb']:.2f} |")
            csv_rows.append([label, r['model'], n_total, f"{r['accuracy']:.4f}", f"{r['precision']:.4f}",
                              f"{r['recall']:.4f}", f"{r['f1']:.4f}", fmt(r['auc']),
                              f"{r['avg_time_s']:.3f}", r['params_b'], r['disk_gb'], r['peak_gpu_gb']])

        if label == 'AEGIS Hard (ham video)' and d3:
            ap_str = ', '.join(f"{g}={d3[g]:.4f}" for g in ('all_fake', 'kling', 'sora') if g in d3)
            print(f"  D3 (NSG-VD/XCLIP-16, ayri metodoloji - AP skoru): {ap_str}")
            md_lines.append(f"\nD3 (NSG-VD/XCLIP-16, ayri metodoloji, AP skoru — dogrudan kiyaslanamaz): {ap_str}\n")
        if label == 'GenBuster-Bench++' and 'genbuster_fake' in d3:
            print(f"  D3 (NSG-VD/XCLIP-16, ayri metodoloji - AP skoru): genbuster_fake={d3['genbuster_fake']:.4f}")
            md_lines.append(f"\nD3 (NSG-VD/XCLIP-16, ayri metodoloji, AP skoru — dogrudan kiyaslanamaz): "
                             f"genbuster_fake={d3['genbuster_fake']:.4f}\n")

    # --- Fake-only datasets (sadece Recall anlamli) ---
    for label, our_json, other_csv, n_total in FAKEONLY_DATASETS:
        rows = []
        if our_json.exists():
            rows.append(load_our_fakeonly(our_json))
        rows += load_other_fakeonly(other_csv)
        rows_sorted = sorted(rows, key=lambda r: -r['recall'])

        print(f"\n{'='*70}")
        print(f"{label} (n={n_total}, SADECE FAKE — Recall = tespit orani)")
        print(f"{'='*70}")
        header = f"{'Model':<26} {'N':<6} {'Recall':<10} {'Sure(s)':<9}"
        print(header)
        print('-' * len(header))
        for r in rows_sorted:
            t = f"{r['avg_time_s']:.3f}" if r['avg_time_s'] is not None else 'N/A'
            print(f"{r['model']:<26} {r['n']:<6} {r['recall']:<10.4f} {t:<9}")

        md_lines.append(f"\n## {label} (n={n_total}, sadece FAKE video — Recall = tespit orani)\n")
        md_lines.append("| Model | N | Recall | Sure(s) |")
        md_lines.append("|---|---|---|---|")
        for r in rows_sorted:
            t = f"{r['avg_time_s']:.3f}" if r['avg_time_s'] is not None else 'N/A'
            md_lines.append(f"| {r['model']} | {r['n']} | {r['recall']:.4f} | {t} |")
            csv_rows.append([label, r['model'], r['n'], '', '', f"{r['recall']:.4f}", '', '',
                              t, '', '', ''])

    OUTPUT_MD.write_text('\n'.join(md_lines) + '\n')
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\nKaydedildi:\n  {OUTPUT_MD}\n  {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
