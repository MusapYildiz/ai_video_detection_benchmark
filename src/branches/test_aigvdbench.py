import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
test_aigvdbench.py - AIGVDBench zip dosyalarini ac ve test et

Kullanim:
    # Once zip'leri ac
    python3 test_aigvdbench.py --extract

    # Sonra test et
    python3 test_aigvdbench.py --test

    # Ikisini birden
    python3 test_aigvdbench.py --extract --test
"""

import os
import sys
import json
import random
import argparse
import zipfile
from pathlib import Path

import torch

THIS_DIR  = f'{BASE_DIR}/ai_video_detector/src/branches'
UTILS_DIR = f'{BASE_DIR}/ai_video_detector/src/utils'
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, UTILS_DIR)

AIGVD_DIR   = Path(f'{BASE_DIR}/datasets/aigvdbench')
EXTRACT_DIR = AIGVD_DIR / 'extracted'
CHECKPOINT  = f'{BASE_DIR}/checkpoints/phase1/checkpoint_best.pt'

# Hangi zip'ler fake, hangisi real
FAKE_ZIPS = [
    'AIGVDBench/ClosedSource/Sora.zip',
    'AIGVDBench/ClosedSource/kling.zip',
    'AIGVDBench/ClosedSource/Gen3.zip',
    'AIGVDBench/ClosedSource/Luma.zip',
    'AIGVDBench/ClosedSource/Gen2.zip',
]

# Real videolar OpenVid-HD'den - ayri zip olmayabilir
# Bunun yerine GenVideo'daki real videolari kullaniriz


def extract_all(n_per_model: int = 100):
    """
    Zip dosyalarini ac, her modelden n_per_model video al.
    Tum zip'i acmak yerine sadece ilk N dosyayi al - hizli.
    """
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    for zip_rel_path in FAKE_ZIPS:
        zip_path = AIGVD_DIR / zip_rel_path
        if not zip_path.exists():
            print(f"[YOK]  {zip_path.name} - henuz indirilmedi, atlaniyor")
            continue

        model_name = zip_path.stem  # Sora, kling, Gen3, ...
        out_dir = EXTRACT_DIR / 'fake' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Kac video var zaten?
        existing = list(out_dir.glob('*.mp4'))
        if len(existing) >= n_per_model:
            print(f"[ATLA] {model_name}: {len(existing)} video zaten var")
            continue

        print(f"[...] {model_name} aciliyor ({zip_path.stat().st_size/1e9:.1f} GB)...")

        extracted = 0
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Sadece mp4 dosyalari al
                mp4_files = [f for f in zf.namelist()
                             if f.lower().endswith('.mp4') and not f.startswith('__')]
                random.shuffle(mp4_files)

                for fname in mp4_files[:n_per_model]:
                    out_path = out_dir / Path(fname).name
                    if out_path.exists():
                        extracted += 1
                        continue
                    with zf.open(fname) as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())
                    extracted += 1

        except Exception as e:
            print(f"  HATA: {e}")
            continue

        print(f"  [OK] {model_name}: {extracted} video acildi → {out_dir}")

    # Toplam
    total = sum(1 for _ in EXTRACT_DIR.rglob('*.mp4'))
    print(f"\nToplam fake video: {total}")


def test_model(n_per_model: int = 50, real_dir: str = None):
    """
    Extract edilmis videolari test et.
    Real videolar icin GenVideo'daki real klasoru kullanilir.
    """
    from detector_model import VideoForensicsDetector
    from video_io import load_video

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Model yukle
    model = VideoForensicsDetector(freeze_dino=True).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    epoch = state.get('epoch', '?')
    auc   = state.get('metrics', {}).get('auc', 0)
    print(f"Checkpoint: epoch={epoch}, val_auc={auc:.4f}\n")

    # Test edilecek kategoriler
    categories = {}

    # Fake - AIGVDBench kapalı kaynak modeller
    fake_base = EXTRACT_DIR / 'fake'
    if fake_base.exists():
        for model_dir in sorted(fake_base.iterdir()):
            if model_dir.is_dir():
                videos = list(model_dir.glob('*.mp4'))
                if videos:
                    categories[f'FAKE/{model_dir.name}'] = {
                        'videos': videos,
                        'label':  1,
                    }

    # Real - GenVideo'daki real videolar
    genvideo_real = Path(f'{BASE_DIR}/datasets/genvideo/extracted/real/msrvtt_youku/Real')
    if genvideo_real.exists():
        categories['REAL/msrvtt_youku'] = {
            'videos': list(genvideo_real.glob('*.mp4')),
            'label':  0,
        }
    elif real_dir and Path(real_dir).exists():
        categories['REAL/custom'] = {
            'videos': list(Path(real_dir).rglob('*.mp4')),
            'label':  0,
        }

    if not categories:
        print("HATA: Hic video bulunamadi. Once --extract calistir.")
        return

    # Test
    print(f"{'Kategori':<25} {'N':<5} {'Dogru':<7} {'Yanlis':<8} {'Ort Skor':<10} {'Accuracy'}")
    print('=' * 65)

    all_results = []

    for cat_name, cat_info in categories.items():
        videos    = cat_info['videos']
        true_label = cat_info['label']
        sample    = random.sample(videos, min(n_per_model, len(videos)))

        correct = wrong = 0
        scores  = []

        for vp in sample:
            try:
                bundle = load_video(str(vp), n_frames=16, n_semantic=8)
                frames = bundle.frames_all.unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(frames)
                prob = out['ai_probability'].item()
                scores.append(prob)
                pred = 1 if prob > 0.5 else 0
                if pred == true_label:
                    correct += 1
                else:
                    wrong += 1
            except Exception:
                wrong += 1

        avg = sum(scores) / len(scores) if scores else 0
        acc = correct / (correct + wrong) if (correct + wrong) > 0 else 0
        n   = correct + wrong

        print(f"{cat_name:<25} {n:<5} {correct:<7} {wrong:<8} {avg:<10.3f} {acc:.1%}")
        all_results.append({
            'category': cat_name,
            'n': n, 'correct': correct, 'wrong': wrong,
            'avg_score': avg, 'accuracy': acc,
        })

    # Ozet
    print('=' * 65)
    fake_results = [r for r in all_results if r['category'].startswith('FAKE')]
    real_results = [r for r in all_results if r['category'].startswith('REAL')]

    if fake_results:
        avg_fake_acc = sum(r['accuracy'] for r in fake_results) / len(fake_results)
        print(f"\nFake modeller ortalama accuracy: {avg_fake_acc:.1%}")
    if real_results:
        avg_real_acc = sum(r['accuracy'] for r in real_results) / len(real_results)
        print(f"Real videolar accuracy:          {avg_real_acc:.1%}")

    # Kaydet
    out_json = AIGVD_DIR / 'test_results.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSonuclar kaydedildi: {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract',      action='store_true', help='Zip dosyalarini ac')
    parser.add_argument('--test',         action='store_true', help='Model testi yap')
    parser.add_argument('--n_per_model',  type=int, default=50,
                        help='Her modelden kac video kullan')
    parser.add_argument('--real_dir',     type=str, default=None,
                        help='Ozel real video klasoru')
    args = parser.parse_args()

    if not args.extract and not args.test:
        parser.print_help()
        sys.exit(0)

    if args.extract:
        extract_all(n_per_model=args.n_per_model)

    if args.test:
        test_model(n_per_model=args.n_per_model, real_dir=args.real_dir)
