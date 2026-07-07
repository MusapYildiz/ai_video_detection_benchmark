import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
build_fresh_real_test.py - Egitimde/val'de KESINLIKLE kullanilmamis
real video test seti olusturur.

Kinetics-400, HD-VG-130M ve MSR-VTT/Youku'dan, train.csv ve val.csv'de
GORULMEMIS videolari secer. Bu, AEGIS'e ek olarak FPR olcumunu
guclendiren ikinci, bagimsiz bir real-video test kaynagidir.

Kullanim:
    python3 build_fresh_real_test.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

BASE = Path(f'{BASE_DIR}')
OUT_CSV = BASE / 'datasets/final_dataset/fresh_real_test.csv'

SOURCES = {
    'kinetics400':   (BASE / 'datasets/kinetics400/videos', 300),
    'hd_vg_130m':    (BASE / 'datasets/genvidbench/extracted/real/hd_vg_130m', 300),
    'msrvtt_youku':  (BASE / 'datasets/genvideo/extracted/real/msrvtt_youku/Real', 300),
}


def load_used_paths():
    used = set()
    for csv_file in ['train.csv', 'val.csv']:
        path = BASE / 'datasets/final_dataset' / csv_file
        with open(path) as f:
            for row in csv.DictReader(f):
                used.add(row['path'])
    return used


def main():
    used_paths = load_used_paths()
    print(f"Egitim/val'de kullanilan toplam video: {len(used_paths)}\n")

    rows = []
    for src, (dir_path, n_target) in SOURCES.items():
        all_videos = list(dir_path.rglob('*.mp4'))
        unused = [v for v in all_videos if str(v) not in used_paths]
        n = min(n_target, len(unused))
        selected = random.sample(unused, n)

        for v in selected:
            rows.append({
                'path': str(v),
                'label': 0,  # hepsi real
                'source': src,
                'difficulty': 'fresh_real_test',
            })

        print(f"{src:<15} diskte={len(all_videos):<7} kullanilmayan={len(unused):<7} secilen={n}")

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'label', 'source', 'difficulty'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nToplam: {len(rows)} video")
    print(f"Kaydedildi: {OUT_CSV}")
    print(f"\nGuvenlik kontrolu: bu videolarin HICBIRI train.csv/val.csv'de yok.")


if __name__ == '__main__':
    main()
