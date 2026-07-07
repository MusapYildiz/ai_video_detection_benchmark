"""
download_dataset.py — GenVideo-100K secici indirme

GenVideo-100K'da her model ayri bir arsivde.
Tum dataset 80-100 GB. Hizli baslamak icin sadece secili modelleri inelim.

Kullanim:
    # Once dosya listesini gör
    python3 download_dataset.py --list

    # Hafif baslangic: Latte (T2V) + MSRVTT (real)
    python3 download_dataset.py --light --output /kullanici_yedek/musap.yildiz/datasets/genvideo

    # Belirli dosyalar
    python3 download_dataset.py --files Latte.tar.gz MSRVTT.zip --output ...

    # Tam dataset
    python3 download_dataset.py --full --output ...
"""

import argparse
import sys
from pathlib import Path


# Hafif baslangic icin secilen dosyalar (~5-10 GB)
LIGHT_FILES = [
    "Real_part_aa",
    "OpenSora.tar.gz",
    "Latte.tar.gz",
    "ZeroScope.tar.gz",
    "pika.tar.gz",
    "SEINE.tar.gz",
    "SVD.tar.gz",
]


def list_files():
    """ModelScope'tan dosya listesini al."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    try:
        files = api.list_repo_files("cccnju/GenVideo-100K", repo_type="dataset")
        print(f"\nGenVideo-100K dosya listesi ({len(files)} dosya):\n")
        for f in sorted(files):
            print(f"  {f}")
        return files
    except Exception as e:
        print(f"Hata: {e}")
        print("\nAlternatif: tarayicidan kontrol et:")
        print("  https://modelscope.cn/datasets/cccnju/GenVideo-100K/files")
        sys.exit(1)


def download_files(output_dir: str, files: list[str] = None, full: bool = False):
    """
    Belirtilen dosyalari indir.

    Args:
        output_dir: indirme klasoru
        files: indirilecek dosyalar (None = tum dataset)
        full: True ise tum dataset
    """
    from modelscope.hub.snapshot_download import snapshot_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if full:
        print(f"TUM dataset indiriliyor (80-100 GB)...")
        print(f"  Hedef: {output_path}")
        print(f"  5-8 saat surebilir.\n")
        kwargs = {}
    else:
        print(f"Secili dosyalar indiriliyor...")
        print(f"  Dosyalar: {files}")
        print(f"  Hedef:    {output_path}\n")
        kwargs = {"allow_patterns": files}

    try:
        local_path = snapshot_download(
            "cccnju/GenVideo-100K",
            cache_dir=str(output_path),
            repo_type="dataset",
            **kwargs,
        )
        print(f"\nIndirme tamamlandi:")
        print(f"  Lokasyon: {local_path}")
        return local_path
    except Exception as e:
        print(f"\nHata: {e}")
        sys.exit(1)


def explore_downloaded(local_path: str):
    """Indirilen dataset yapisini goster."""
    local_path = Path(local_path)
    if not local_path.exists():
        print(f"Klasor yok: {local_path}")
        return

    print(f"\n{'='*50}")
    print(f"Dataset: {local_path}")
    print(f"{'='*50}\n")

    for item in sorted(local_path.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.name}  ({size_mb:.1f} MB)")
        elif item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            size_mb = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / 1e6
            print(f"  {item.name}/  ({n_files} dosya, {size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="/kullanici_yedek/musap.yildiz/datasets/genvideo")
    parser.add_argument("--list", action="store_true",
                        help="Sadece dosya listesini goster")
    parser.add_argument("--light", action="store_true",
                        help="Hafif baslangic (~5-10 GB)")
    parser.add_argument("--full", action="store_true",
                        help="Tum dataset (~80-100 GB)")
    parser.add_argument("--files", nargs="+",
                        help="Belirli dosyalari indir")
    parser.add_argument("--explore-only", action="store_true",
                        help="Indirmeden mevcut klasoru kesfet")
    args = parser.parse_args()

    if args.list:
        list_files()
    elif args.explore_only:
        explore_downloaded(args.output)
    elif args.light:
        download_files(args.output, files=LIGHT_FILES)
    elif args.files:
        download_files(args.output, files=args.files)
    elif args.full:
        download_files(args.output, full=True)
    else:
        print("Hangi modu istiyorsun?")
        print("  --list           Dosya listesi")
        print("  --light          Hafif baslangic (~5-10 GB)")
        print("  --files X Y      Belirli dosyalar")
        print("  --full           Tum dataset")
        sys.exit(0)
