"""
AEGIS Hard Test Set - Ground Truth Manifest Olusturucu
----------------------------------------------------------
Klasor yapisindan (test_data/real/youtube/*.mp4, test_data/kling/*.mp4,
test_data/sora/*.mp4) ground truth bilgisini cikarip tek bir CSV manifest
uretir. Bu manifest, 6 modelin sonuclarini karsilastirmak icin ortak referans
olarak kullanilacak.

Cikti kolonlari:
    video_path, ground_truth (real/fake), generator (camera/kling/sora)

Kullanim:
    python build_manifest.py --root $AEGIS_BASE_DIR/datasets/aegis_full/videos/test_data \
        --output manifest.csv
"""

import argparse
import csv
from pathlib import Path

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

# Klasor adi -> (ground_truth, generator) eslemesi
# 'real' altindaki her alt klasor (youtube vb.) gercek video, generator='camera'
# diger ust-seviye klasorler (kling, sora) fake, generator = klasor adi
def classify(video_path: Path, root: Path):
    rel = video_path.relative_to(root)
    parts = rel.parts
    top = parts[0].lower()
    if top == 'real':
        # real/youtube/*.mp4, real/dvf/*.mp4 -> hepsi gercek video, generator='camera'
        return 'real', 'camera'
    elif top == 'ai_gen' and len(parts) > 1:
        # ai_gen/kling/*.mp4, ai_gen/sora/*.mp4 -> generator ikinci seviyeden okunur
        return 'fake', parts[1].lower()
    else:
        # beklenmeyen bir yapi - ust seviye klasor adini generator olarak kullan
        return 'fake', top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                         help="AEGIS test_data kok dizini (real/, kling/, sora/ alt klasorlerini icerir)")
    parser.add_argument('--output', type=str, default='manifest.csv')
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for vp in sorted(root.rglob('*')):
        if vp.is_file() and vp.suffix.lower() in VIDEO_EXTS:
            gt, generator = classify(vp, root)
            rows.append([str(vp), gt, generator])

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['video_path', 'ground_truth', 'generator'])
        writer.writerows(rows)

    # Ozet
    from collections import Counter
    gt_counts = Counter(r[1] for r in rows)
    gen_counts = Counter(r[2] for r in rows)
    print(f"[info] Toplam {len(rows)} video bulundu, manifest kaydedildi: {args.output}")
    print(f"[info] Ground truth dagilimi: {dict(gt_counts)}")
    print(f"[info] Generator dagilimi: {dict(gen_counts)}")


if __name__ == '__main__':
    main()
