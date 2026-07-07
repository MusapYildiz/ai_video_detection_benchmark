import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
build_benchmark_table.py - Tum modellerin AEGIS sonuclarini birlestirip
tek bir nihai karsilastirma tablosu uretir.

Kaynaklar:
  1. Bizim model: ablation_aegis.json (ablation_eval.py ciktisi)
  2. Diger 5 model: comparison_report.csv (compute_metrics.py ciktisi)

Kullanim:
    python3 build_benchmark_table.py
"""

import csv
import json
from pathlib import Path

OUR_MODEL_JSON      = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_aegis_raw.json')
OUR_EXTRA_TEST_JSON = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_extra_test.json')
OUR_AIGVD_JSON      = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_aigvdbench.json')
OUR_GENBUSTER_JSON = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_genbuster.json')
COMPARISON_CSV = Path(f'{BASE_DIR}/ai_video_detector/other_models/comparison_report.csv')
D3_RESULTS_DIR = Path(f'{BASE_DIR}/ai_video_detector/other_models/D3/results')
OUTPUT_MD       = Path(f'{BASE_DIR}/checkpoints/phase2/benchmark_table.md')
OUTPUT_CSV      = Path(f'{BASE_DIR}/checkpoints/phase2/benchmark_table.csv')


def load_our_model():
    """ablation_aegis.json'dan bizim modelin metriklerini cikarir."""
    with open(OUR_MODEL_JSON) as f:
        d = json.load(f)

    ana = d['metrics']['pixel+motion+consistency (ana)']
    perf = d['performance']
    info = d['model_info']
    gen = d.get('generator_breakdown', {})

    row = {
        'model':       'Bizim Model (ham video)',
        'accuracy':    ana['accuracy'],
        'precision':   ana['precision'],
        'recall':      ana['recall'],
        'f1':          ana['f1'],
        'auc':         ana['auc'],
        'avg_time_s':  perf['avg_time_per_video_s'],
        'params_b':    info['n_params_total'] / 1e9,
        'disk_gb':     info['checkpoint_size_mb'] / 1024,
        'peak_gpu_gb': perf['peak_gpu_memory_mb'] / 1024 if perf.get('peak_gpu_memory_mb') else None,
        'camera_acc':  gen.get('camera', {}).get('accuracy'),
        'kling_acc':   gen.get('kling', {}).get('accuracy'),
        'sora_acc':    gen.get('sora', {}).get('accuracy'),
    }
    return row


def load_other_models():
    """
    comparison_report.csv'den diger modellerin satirlarini okur.

    CSV YAPISI: Her model icin 4 satir var (category: camera, kling,
    sora, 'TOPLU (ALL)'). Ana tabloya sadece 'TOPLU (ALL)' satirini
    aliyoruz, camera/kling/sora satirlarini generator_breakdown icin
    ayri ayri topluyoruz.
    """
    rows = []
    breakdowns = {}  # model -> {camera: acc, kling: acc, sora: acc}

    if not COMPARISON_CSV.exists():
        print(f"[UYARI] {COMPARISON_CSV} bulunamadi, sadece bizim model dahil edilecek.")
        return rows

    with open(COMPARISON_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            model = r['model']
            category = r['category']

            if category in ('camera', 'kling', 'sora'):
                breakdowns.setdefault(model, {})[category] = float(r['accuracy'])
                continue

            if category != 'TOPLU (ALL)':
                continue

            auc_val = r.get('auc', '')
            rows.append({
                'model':       model,
                'accuracy':    float(r['accuracy']),
                'precision':   float(r['precision']),
                'recall':      float(r['recall']),
                'f1':          float(r['f1']),
                'auc':         float(auc_val) if auc_val not in (None, '', 'N/A') else None,
                'avg_time_s':  float(r['mean_seconds']),
                'params_b':    float(r['param_count_billions']),
                'disk_gb':     float(r['disk_size_gb']),
                'peak_gpu_gb': float(r['peak_gpu_memory_gb']),
                'camera_acc':  None,  # asagida breakdowns'tan doldurulacak
                'kling_acc':   None,
                'sora_acc':    None,
            })

    # Generator breakdown'i ilgili satira isle
    for row in rows:
        b = breakdowns.get(row['model'], {})
        row['camera_acc'] = b.get('camera')
        row['kling_acc']  = b.get('kling')
        row['sora_acc']   = b.get('sora')

    return rows


def load_our_extra_sources():
    """
    extra_test.json ve aigvdbench.json'dan bizim modelin
    ek test kaynaklarindaki performansini cikarir.

    Bu kaynaklar SADECE FAKE video icerdigi icin AUC ve FPR
    anlamsiz (n_real=0). Anlamli metrik: Recall (AI videolarini
    yakalama orani).
    """
    sources = {}

    for name, path in [('extra_test (Sora)', OUR_EXTRA_TEST_JSON),
                       ('aigvdbench',         OUR_AIGVD_JSON)]:
        if not path.exists():
            print(f"[UYARI] {path} bulunamadi, atlaniyor.")
            continue
        with open(path) as f:
            d = json.load(f)

        ana = d['metrics'].get('pixel+motion+consistency (ana)', {})
        perf = d.get('performance', {})
        gen_breakdown = d.get('generator_breakdown', {})

        sources[name] = {
            'accuracy':     ana.get('accuracy'),
            'recall':       ana.get('recall'),
            'f1':           ana.get('f1'),
            'avg_time_s':   perf.get('avg_time_per_video_s'),
            'n':            ana.get('n'),
            'n_fake':       ana.get('n_fake'),
            'generators':   {g: v.get('accuracy') for g, v in gen_breakdown.items()},
        }

    return sources


def print_extra_sources_section(sources):
    if not sources:
        return

    print(f"\n{'='*90}")
    print("EK TEST KAYNAKLARI — Bizim Model (sadece fake video, FPR hesaplanamaz)")
    print(f"{'='*90}")
    print("Not: Bu kaynaklar yalnizca AI uretimi video iceriyor (real karsiligi yok).")
    print("     Buradaki anlamli metrik Recall'dur (AI videolarini yakalama orani).\n")
    print(f"{'Kaynak':<25} {'N':<6} {'Accuracy':<10} {'Recall':<10} {'F1':<10} {'Sure(s)':<10}")
    print('-' * 75)
    for name, s in sources.items():
        acc  = f"{s['accuracy']:.4f}"  if s['accuracy']  is not None else "-"
        rec  = f"{s['recall']:.4f}"    if s['recall']     is not None else "-"
        f1   = f"{s['f1']:.4f}"        if s['f1']         is not None else "-"
        t    = f"{s['avg_time_s']:.3f}" if s['avg_time_s'] is not None else "-"
        print(f"{name:<25} {s['n']:<6} {acc:<10} {rec:<10} {f1:<10} {t:<10}")

        if s['generators']:
            for gen, acc_g in sorted(s['generators'].items()):
                gen_acc = f"{acc_g:.4f}" if acc_g is not None else "-"
                print(f"  └ {gen:<22} {'':>6} {gen_acc}")

    # Markdown'a ekle
    with open(OUTPUT_MD, 'a') as f:
        f.write("\n## Ek Test Kaynaklari (Bizim Model — Sadece Fake Video)\n\n")
        f.write("Not: Bu kaynaklar yalnizca AI uretimi video iceriyor, FPR hesaplanamaz. "
                "Anlamli metrik Recall'dur.\n\n")
        f.write("| Kaynak | N | Accuracy | Recall | F1 | Sure(s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, s in sources.items():
            acc = f"{s['accuracy']:.4f}" if s['accuracy'] is not None else "-"
            rec = f"{s['recall']:.4f}"   if s['recall']   is not None else "-"
            f1  = f"{s['f1']:.4f}"       if s['f1']       is not None else "-"
            t   = f"{s['avg_time_s']:.3f}" if s['avg_time_s'] is not None else "-"
            f.write(f"| {name} | {s['n']} | {acc} | {rec} | {f1} | {t} |\n")

        # Generator bazinda kirinim
        f.write("\n### Generator Bazinda Accuracy (Ek Kaynaklar)\n\n")
        f.write("| Kaynak | Generator | Accuracy |\n")
        f.write("|---|---|---|\n")
        for name, s in sources.items():
            for gen, acc_g in sorted(s['generators'].items()):
                gen_acc = f"{acc_g:.4f}" if acc_g is not None else "-"
                f.write(f"| {name} | {gen} | {gen_acc} |\n")


def load_d3_results():
    """
    D3/results/*.txt dosyalarini okur (AP-tabanli, farkli format).
    Her dosya tek bir 'real vs tek-generator' karsilastirmasi icerir.
    Sadece AP skoru var - Accuracy/F1/sure/model-boyutu yok, bu yuzden
    ana tabloya degil, ayri bir 'AP Skorlari' bolumune yazilir.
    """
    results = {}
    if not D3_RESULTS_DIR.exists():
        print(f"[UYARI] {D3_RESULTS_DIR} bulunamadi, D3 atlanacak.")
        return results

    for txt_file in sorted(D3_RESULTS_DIR.glob('*.txt')):
        with open(txt_file) as f:
            content = f.read()

        fake_csv = None
        ap_score = None
        for line in content.splitlines():
            if line.startswith('Fake CSV:'):
                fake_csv = line.split(':', 1)[1].strip()
            elif line.startswith('AP Score:'):
                ap_score = float(line.split(':', 1)[1].strip())

        if fake_csv and ap_score is not None:
            # 'datasets/csv/kling.csv' -> 'kling'
            generator = Path(fake_csv).stem
            results[generator] = ap_score

    return results


def print_d3_section(d3_results):
    if not d3_results:
        return
    print(f"\n{'='*50}")
    print("D3 (NSG-VD / XCLIP-16) — AP SKORLARI")
    print(f"{'='*50}")
    print("Not: D3 farkli bir metodoloji kullanir (Average Precision,")
    print("     real vs TEK generator ikili karsilastirma). Accuracy/F1/")
    print("     sure/model-boyutu bilgisi yok, bu yuzden ana tabloya")
    print("     dahil edilmemistir.\n")
    for gen, ap in sorted(d3_results.items()):
        print(f"  Real vs {gen:<10} AP Score: {ap:.4f}")

    with open(OUTPUT_MD, 'a') as f:
        f.write("\n## D3 (NSG-VD / XCLIP-16) — AP Skorlari\n\n")
        f.write("Not: Farkli metodoloji (Average Precision, real vs tek generator ikili "
                "karsilastirma). Diger modellerle dogrudan ayni sutunlarda kiyaslanamaz.\n\n")
        f.write("| Real vs Generator | AP Score |\n")
        f.write("|---|---|\n")
        for gen, ap in sorted(d3_results.items()):
            f.write(f"| {gen} | {ap:.4f} |\n")


def print_and_save(rows):
    # Genel tablo - buyukten kucuge AUC'ye gore sirala (None'lar sona)
    rows_sorted = sorted(rows, key=lambda r: (r['auc'] is None, -(r['auc'] or 0)))

    print(f"\n{'='*125}")
    print("NIHAI BENCHMARK TABLOSU — AEGIS Hard Test Set (n=436)")
    print(f"{'='*125}")
    header = f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'AUC':<8} {'Sure(s)':<10} {'Param(B)':<10} {'Disk(GB)':<10} {'GPU(GB)':<10}"
    print(header)
    print('-' * 125)
    for r in rows_sorted:
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
        print(f"{r['model']:<15} {r['accuracy']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} "
              f"{r['f1']:<8.4f} {auc_str:<8} {r['avg_time_s']:<10.3f} {r['params_b']:<10.3f} "
              f"{r['disk_gb']:<10.2f} {r['peak_gpu_gb']:<10.2f}")

    # Generator bazinda (sadece bilgisi olanlar)
    print(f"\n{'='*60}")
    print("GENERATOR BAZINDA ACCURACY (varsa)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Camera':<10} {'Kling':<10} {'Sora':<10}")
    print('-' * 60)
    for r in rows_sorted:
        cam = f"{r['camera_acc']:.4f}" if r['camera_acc'] is not None else "-"
        kli = f"{r['kling_acc']:.4f}" if r['kling_acc'] is not None else "-"
        sor = f"{r['sora_acc']:.4f}" if r['sora_acc'] is not None else "-"
        print(f"{r['model']:<15} {cam:<10} {kli:<10} {sor:<10}")

    # Markdown kaydet
    with open(OUTPUT_MD, 'w') as f:
        f.write("# AEGIS Hard Test Set — Benchmark Karsilastirma Tablosu\n\n")
        f.write(f"Toplam ornek: 436 (218 real + 111 kling + 107 sora)\n\n")
        f.write("## Genel Performans\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 | AUC | Sure(s) | Param(B) | Disk(GB) | GPU(GB) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows_sorted:
            auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "N/A"
            f.write(f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
                    f"{r['recall']:.4f} | {r['f1']:.4f} | {auc_str} | {r['avg_time_s']:.3f} | "
                    f"{r['params_b']:.3f} | {r['disk_gb']:.2f} | {r['peak_gpu_gb']:.2f} |\n")

        f.write("\n## Generator Bazinda Accuracy\n\n")
        f.write("| Model | Camera (Real) | Kling | Sora |\n")
        f.write("|---|---|---|---|\n")
        for r in rows_sorted:
            cam = f"{r['camera_acc']:.4f}" if r['camera_acc'] is not None else "-"
            kli = f"{r['kling_acc']:.4f}" if r['kling_acc'] is not None else "-"
            sor = f"{r['sora_acc']:.4f}" if r['sora_acc'] is not None else "-"
            f.write(f"| {r['model']} | {cam} | {kli} | {sor} |\n")

    # CSV kaydet
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                      'avg_time_s', 'params_b', 'disk_gb', 'peak_gpu_gb',
                      'camera_acc', 'kling_acc', 'sora_acc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"\nKaydedildi:")
    print(f"  {OUTPUT_MD}")
    print(f"  {OUTPUT_CSV}")


def main():
    our_row = load_our_model()
    other_rows = load_other_models()
    all_rows = [our_row] + other_rows
    print_and_save(all_rows)

    extra_sources = load_our_extra_sources()
    print_extra_sources_section(extra_sources)

    d3_results = load_d3_results()
    print_d3_section(d3_results)


if __name__ == '__main__':
    main()
