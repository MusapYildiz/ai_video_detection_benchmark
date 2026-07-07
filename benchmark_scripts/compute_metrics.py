"""
AEGIS Hard Test Set - Metrik Hesaplama ve Model Karsilastirma
-------------------------------------------------------------------
Her modelin kendi infer.py'sinden urettigi sonuc dosyasini (CSV ya da JSON)
manifest.csv (ground truth) ile birlestirir, generator-bazli (camera/kling/sora)
ve toplu (real vs fake) metrikleri hesaplar, sonunda 6 modeli karsilastiran
tek bir ozet tablo basar.

NOT: D3 bu scripte dahil DEGIL - D3'un cikisi (AP score) per-video binary karar
degil, ayri bir degerlendirme paradigmasi (average precision uzerinden). D3'u
ayri raporlayacagiz.

Desteklenen model cikti formatlari (otomatik algilanir):
    CSV  - video_path, ..., verdict/prediction (REAL veya FAKE icerebilir)
    JSON - video_path, ..., answer (Real veya Fake icerebilir) [Skyra formati]

Kullanim:
    python compute_metrics.py --manifest manifest.csv \
        --result cocovideo:CoCoVideo/results.csv \
        --result videoveritas:VideoVeritas/results.csv \
        --result skyra:Skyra/eval/inference_end2end/results.json \
        --result ivyfake:IvyFake/results.csv \
        --result busterx:BusterX/results.csv \
        --output comparison_report.csv
"""

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict


def normalize_verdict(raw):
    """Herhangi bir model ciktisini REAL/FAKE/UNKNOWN'a normalize eder."""
    if raw is None:
        return "UNKNOWN"
    s = str(raw).strip().upper()
    if "REAL" in s:
        return "REAL"
    if "FAKE" in s:
        return "FAKE"
    return "UNKNOWN"


def load_model_results(path):
    """CSV ya da JSON sonuc dosyasini {video_path: (verdict, inference_seconds, fake_probability)} sozlugune cevirir."""
    path = Path(path)
    results = {}

    def _parse_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _extract_fake_prob(row):
        # Once dogrudan fake_probability kolonunu dene (transformers-tabanli 5 model)
        fp = _parse_float(row.get('fake_probability'))
        if fp is not None:
            return fp
        # CoCoVideo'da confidence_score = P(REAL) -> P(FAKE) = 1 - score
        cs = _parse_float(row.get('confidence_score'))
        if cs is not None:
            return 1.0 - cs
        return None

    if path.suffix.lower() == '.json':
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        for row in data:
            vp = row.get('video_path')
            verdict = normalize_verdict(row.get('answer') or row.get('verdict') or row.get('prediction'))
            seconds = _parse_float(row.get('inference_seconds'))
            fake_prob = _extract_fake_prob(row)
            if vp:
                results[str(Path(vp).resolve())] = (verdict, seconds, fake_prob)

    else:  # CSV
        with open(path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = row.get('video_path')
                verdict = normalize_verdict(
                    row.get('verdict') or row.get('prediction') or row.get('answer')
                )
                seconds = _parse_float(row.get('inference_seconds'))
                fake_prob = _extract_fake_prob(row)
                if vp:
                    results[str(Path(vp).resolve())] = (verdict, seconds, fake_prob)

    return results


def load_manifest(path):
    """video_path -> (ground_truth, generator) sozlugu."""
    manifest = {}
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vp = str(Path(row['video_path']).resolve())
            gt = "FAKE" if row['ground_truth'].strip().lower() == 'fake' else "REAL"
            manifest[vp] = (gt, row['generator'])
    return manifest


def compute_auc(y_true, y_score):
    """Mann-Whitney U tabanli AUC hesabi (sklearn bagimliligi olmadan).
    y_true: 1=FAKE, 0=REAL.  y_score: P(fake) tahmini."""
    pairs = [(t, s) for t, s in zip(y_true, y_score) if s is not None]
    pos = [s for t, s in pairs if t == 1]
    neg = [s for t, s in pairs if t == 0]
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


def compute_confusion(pairs):
    """pairs: list of (ground_truth, prediction) -> dict with tp/tn/fp/fn/accuracy/precision/recall/f1."""
    tp = sum(1 for gt, pred in pairs if gt == "FAKE" and pred == "FAKE")
    tn = sum(1 for gt, pred in pairs if gt == "REAL" and pred == "REAL")
    fp = sum(1 for gt, pred in pairs if gt == "REAL" and pred == "FAKE")
    fn = sum(1 for gt, pred in pairs if gt == "FAKE" and pred == "REAL")
    unknown = sum(1 for gt, pred in pairs if pred == "UNKNOWN")
    total = len(pairs)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        'n': total, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'unknown': unknown,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
    }


def compute_timing_stats(seconds_list):
    seconds_list = [s for s in seconds_list if s is not None]
    if not seconds_list:
        return None
    return {
        'n_timed': len(seconds_list),
        'mean': statistics.mean(seconds_list),
        'median': statistics.median(seconds_list),
        'min': min(seconds_list),
        'max': max(seconds_list),
        'total': sum(seconds_list),
    }


def evaluate_model(model_name, results, manifest):
    """Generator-bazli ve toplu metrikleri + sure istatistikleri + AUC hesaplar."""
    by_generator = defaultdict(list)   # generator -> [(gt, pred), ...]
    timing_by_generator = defaultdict(list)
    score_by_generator = defaultdict(list)   # generator -> [(y_true 0/1, fake_prob), ...]
    all_pairs = []
    all_timings = []
    all_scores = []
    missing = 0

    for vp, (gt, generator) in manifest.items():
        entry = results.get(vp)
        if entry is None:
            missing += 1
            continue
        pred, seconds, fake_prob = entry
        by_generator[generator].append((gt, pred))
        timing_by_generator[generator].append(seconds)
        y_true_bin = 1 if gt == "FAKE" else 0
        score_by_generator[generator].append((y_true_bin, fake_prob))
        all_pairs.append((gt, pred))
        all_timings.append(seconds)
        all_scores.append((y_true_bin, fake_prob))

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    if missing:
        print(f"[uyari] Manifest'te olan ama bu modelde sonucu olmayan video sayisi: {missing}")

    rows = []
    for generator in sorted(by_generator.keys()):
        m = compute_confusion(by_generator[generator])
        t = compute_timing_stats(timing_by_generator[generator])
        y_true_list, y_score_list = zip(*score_by_generator[generator])
        auc = compute_auc(list(y_true_list), list(y_score_list))
        m['timing'] = t
        m['auc'] = auc
        rows.append((generator, m))
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"\n  -- Generator: {generator} (n={m['n']}) --")
        print(f"     Accuracy: {m['accuracy']:.4f}  Precision: {m['precision']:.4f}  "
              f"Recall: {m['recall']:.4f}  F1: {m['f1']:.4f}  AUC: {auc_str}")
        print(f"     TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} Unknown={m['unknown']}")
        if t:
            print(f"     Sure (s): ortalama={t['mean']:.2f}  medyan={t['median']:.2f}  "
                  f"min={t['min']:.2f}  max={t['max']:.2f}  toplam={t['total']:.1f}")

    overall = compute_confusion(all_pairs)
    overall_timing = compute_timing_stats(all_timings)
    y_true_all, y_score_all = zip(*all_scores)
    overall_auc = compute_auc(list(y_true_all), list(y_score_all))
    overall['timing'] = overall_timing
    overall['auc'] = overall_auc
    rows.append(('TOPLU (ALL)', overall))
    auc_str = f"{overall_auc:.4f}" if overall_auc is not None else "N/A"
    print(f"\n  -- TOPLU (tum generatorlar) (n={overall['n']}) --")
    print(f"     Accuracy: {overall['accuracy']:.4f}  Precision: {overall['precision']:.4f}  "
          f"Recall: {overall['recall']:.4f}  F1: {overall['f1']:.4f}  AUC: {auc_str}")
    print(f"     TP={overall['tp']} TN={overall['tn']} FP={overall['fp']} FN={overall['fn']} "
          f"Unknown={overall['unknown']}")
    if overall_timing:
        print(f"     Sure (s): ortalama={overall_timing['mean']:.2f}  medyan={overall_timing['median']:.2f}  "
              f"min={overall_timing['min']:.2f}  max={overall_timing['max']:.2f}  "
              f"toplam={overall_timing['total']:.1f} ({overall_timing['total']/60:.1f} dk)")

    return rows


def evaluate_model_fakeonly(model_name, results, manifest):
    """Sadece fake video iceren datasetler icin recall (tespit orani) hesaplar.
    AUC, accuracy, precision hesaplanamaz (real video yok)."""
    by_generator = defaultdict(list)
    timing_by_generator = defaultdict(list)
    score_by_generator = defaultdict(list)
    all_pairs = []
    all_timings = []
    all_scores = []
    missing = 0

    for vp, (gt, generator) in manifest.items():
        entry = results.get(vp)
        if entry is None:
            missing += 1
            continue
        pred, seconds, fake_prob = entry
        by_generator[generator].append(pred)
        timing_by_generator[generator].append(seconds)
        score_by_generator[generator].append(fake_prob)
        all_pairs.append(pred)
        all_timings.append(seconds)
        all_scores.append(fake_prob)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} (FAKE-ONLY MODE)")
    print(f"{'='*70}")
    if missing:
        print(f"[uyari] Manifest'te olan ama bu modelde sonucu olmayan: {missing}")

    def recall_stats(preds):
        total = len(preds)
        detected = sum(1 for p in preds if p == 'FAKE')
        unknown = sum(1 for p in preds if p == 'UNKNOWN')
        missed = sum(1 for p in preds if p == 'REAL')
        recall = detected / total if total else 0.0
        return {'n': total, 'detected': detected, 'missed': missed,
                'unknown': unknown, 'recall': recall}

    rows = []
    for generator in sorted(by_generator.keys()):
        m = recall_stats(by_generator[generator])
        t = compute_timing_stats(timing_by_generator[generator])
        m['timing'] = t
        rows.append((generator, m))
        print(f"\n  -- Generator: {generator} (n={m['n']}) --")
        print(f"     Recall (tespit orani): {m['recall']:.4f}  "
              f"Detected={m['detected']} Missed={m['missed']} Unknown={m['unknown']}")
        if t:
            print(f"     Sure (s): ortalama={t['mean']:.2f}  medyan={t['median']:.2f}  "
                  f"toplam={t['total']:.1f}")

    overall = recall_stats(all_pairs)
    overall_timing = compute_timing_stats(all_timings)
    overall['timing'] = overall_timing
    rows.append(('TOPLU (ALL)', overall))
    print(f"\n  -- TOPLU (n={overall['n']}) --")
    print(f"     Recall (tespit orani): {overall['recall']:.4f}  "
          f"Detected={overall['detected']} Missed={overall['missed']} Unknown={overall['unknown']}")
    if overall_timing:
        print(f"     Sure (s): ortalama={overall_timing['mean']:.2f}  "
              f"toplam={overall_timing['total']:.1f} ({overall_timing['total']/60:.1f} dk)")

    return rows
    """model_info.json dosyasini okur (param_count, disk_size_gb, peak_gpu_memory_gb)."""
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_model_info(path):
    """model_info.json dosyasini okur."""
    try:
        import json
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, Exception):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--result', type=str, action='append', required=True,
                         help="model_adi:sonuc_dosyasi.csv (ya da .json) formatinda, tekrar tekrar verilebilir")
    parser.add_argument('--model_info', type=str, action='append', default=[],
                         help="model_adi:model_info.json formatinda (parametre sayisi/disk/GPU bellegi icin), opsiyonel")
    parser.add_argument('--output', type=str, default='comparison_report.csv')
    parser.add_argument('--fakeonly', action='store_true',
                         help="Sadece fake video iceren datasetler icin: sadece recall hesaplar")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"[info] Manifest yuklendi: {len(manifest)} video")

    model_infos = {}
    for spec in args.model_info:
        if ':' not in spec:
            sys.exit(f"Gecersiz --model_info formati: {spec}")
        model_name, info_path = spec.split(':', 1)
        info = load_model_info(info_path)
        if info:
            model_infos[model_name] = info

    all_model_rows = {}
    for spec in args.result:
        if ':' not in spec:
            sys.exit(f"Gecersiz --result formati: {spec} (model_adi:dosya_yolu olmali)")
        model_name, result_path = spec.split(':', 1)
        results = load_model_results(result_path)
        print(f"[info] {model_name}: {len(results)} sonuc yuklendi ({result_path})")
        if args.fakeonly:
            rows = evaluate_model_fakeonly(model_name, results, manifest)
        else:
            rows = evaluate_model(model_name, results, manifest)
        all_model_rows[model_name] = dict((r[0], r[1]) for r in rows)

    # Final karsilastirma tablosu
    print(f"\n\n{'='*130}")
    print("FINAL KARSILASTIRMA TABLOSU")
    print(f"{'='*130}")

    if args.fakeonly:
        header = (f"{'Model':<18} {'Recall':>8} {'Detected':>9} {'Missed':>7} {'Unknown':>8} "
                  f"{'Ort.Sure(s)':>11} {'Param(B)':>9} {'Disk(GB)':>9} {'PeakGPU(GB)':>11}")
        print(header)
        print('-' * len(header))

        csv_rows = [['model', 'category', 'n', 'recall', 'detected', 'missed', 'unknown',
                     'mean_seconds', 'median_seconds', 'min_seconds', 'max_seconds', 'total_seconds',
                     'param_count', 'param_count_billions', 'disk_size_gb', 'peak_gpu_memory_gb']]

        for model_name, cat_rows in all_model_rows.items():
            overall = cat_rows.get('TOPLU (ALL)')
            info = model_infos.get(model_name, {})
            if overall:
                t = overall.get('timing')
                mean_s = f"{t['mean']:.2f}" if t else "N/A"
                param_b = info.get('param_count_billions')
                disk_gb = info.get('disk_size_gb')
                peak_gpu = info.get('peak_gpu_memory_gb')
                print(f"{model_name:<18} {overall['recall']:>8.4f} {overall['detected']:>9} "
                      f"{overall['missed']:>7} {overall['unknown']:>8} {mean_s:>11} "
                      f"{param_b if param_b is not None else 'N/A':>9} "
                      f"{disk_gb if disk_gb is not None else 'N/A':>9} "
                      f"{peak_gpu if peak_gpu is not None else 'N/A':>11}")
            for cat, m in cat_rows.items():
                t = m.get('timing')
                csv_rows.append([
                    model_name, cat, m['n'], f"{m['recall']:.4f}", m['detected'], m['missed'], m['unknown'],
                    f"{t['mean']:.2f}" if t else '', f"{t['median']:.2f}" if t else '',
                    f"{t['min']:.2f}" if t else '', f"{t['max']:.2f}" if t else '', f"{t['total']:.1f}" if t else '',
                    info.get('param_count', ''), info.get('param_count_billions', ''),
                    info.get('disk_size_gb', ''), info.get('peak_gpu_memory_gb', ''),
                ])
    else:
        header = (f"{'Model':<18} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8} "
                  f"{'Ort.Sure(s)':>11} {'Param(B)':>9} {'Disk(GB)':>9} {'PeakGPU(GB)':>11}")
        print(header)
        print('-' * len(header))

        csv_rows = [['model', 'category', 'n', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'tp', 'tn', 'fp', 'fn',
                     'unknown', 'mean_seconds', 'median_seconds', 'min_seconds', 'max_seconds', 'total_seconds',
                     'param_count', 'param_count_billions', 'disk_size_gb', 'peak_gpu_memory_gb']]

        for model_name, cat_rows in all_model_rows.items():
            overall = cat_rows.get('TOPLU (ALL)')
            info = model_infos.get(model_name, {})
            if overall:
                t = overall.get('timing')
                auc = overall.get('auc')
                mean_s = f"{t['mean']:.2f}" if t else "N/A"
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                param_b = info.get('param_count_billions')
                disk_gb = info.get('disk_size_gb')
                peak_gpu = info.get('peak_gpu_memory_gb')
                print(f"{model_name:<18} {overall['accuracy']:>9.4f} {overall['precision']:>10.4f} "
                      f"{overall['recall']:>8.4f} {overall['f1']:>8.4f} {auc_str:>8} {mean_s:>11} "
                      f"{param_b if param_b is not None else 'N/A':>9} "
                      f"{disk_gb if disk_gb is not None else 'N/A':>9} "
                      f"{peak_gpu if peak_gpu is not None else 'N/A':>11}")
            for cat, m in cat_rows.items():
                t = m.get('timing')
                auc = m.get('auc')
                csv_rows.append([
                    model_name, cat, m['n'], f"{m['accuracy']:.4f}", f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}", f"{m['f1']:.4f}", f"{auc:.4f}" if auc is not None else '',
                    m['tp'], m['tn'], m['fp'], m['fn'], m['unknown'],
                    f"{t['mean']:.2f}" if t else '', f"{t['median']:.2f}" if t else '',
                    f"{t['min']:.2f}" if t else '', f"{t['max']:.2f}" if t else '', f"{t['total']:.1f}" if t else '',
                    info.get('param_count', ''), info.get('param_count_billions', ''),
                    info.get('disk_size_gb', ''), info.get('peak_gpu_memory_gb', ''),
                ])

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\n[info] Detayli rapor kaydedildi: {args.output}")


if __name__ == '__main__':
    main()
