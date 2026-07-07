import os
# Kendi sunucunda calistirmak icin: export AEGIS_BASE_DIR=/senin/yolun
BASE_DIR = os.environ.get("AEGIS_BASE_DIR", "/kullanici_yedek/musap.yildiz")

"""
ablation_eval.py - 7 kombinasyonlu ablation degerlendirmesi

checkpoint_best.pt'yi yukler, asagidaki yedi kombinasyonun
AUC / FPR / F1 / Accuracy degerlerini cikartir:

  Tekil:  pixel, motion, consistency
  Ikili:  pixel+motion, pixel+consistency, motion+consistency
  Uclu:   pixel+motion+consistency (ana fusion)

Iki veri kaynagi uzerinde calisabilir:
  --source val    : final_dataset/preprocessed/val/index.csv (varsayilan)
  --source aegis   : AEGIS Hard Test Set (parquet, keyframe tabanli)

Kullanim:
    CUDA_VISIBLE_DEVICES=4 python3 ablation_eval.py --source val
    CUDA_VISIBLE_DEVICES=4 python3 ablation_eval.py --source aegis
"""

import os
import sys
import csv
import json
import time
import signal
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_model import VideoForensicsDetector

CHECKPOINT = f'{BASE_DIR}/checkpoints/phase2/checkpoint_best.pt'
VAL_INDEX  = f'{BASE_DIR}/datasets/final_dataset/preprocessed/val/index.csv'
AEGIS_DIR  = f'{BASE_DIR}/datasets/aegis/data/'
AEGIS_RAW_VIDEO_DIR = f'{BASE_DIR}/datasets/aegis_full/videos/test_data'
GENBUSTER_DIR       = f'{BASE_DIR}/datasets/GenBuster-Bench-plusplus/video'
EXTRA_TEST_CSV = f'{BASE_DIR}/datasets/final_dataset/extra_test.csv'
FRESH_REAL_TEST_CSV = f'{BASE_DIR}/datasets/final_dataset/fresh_real_test.csv'
AIGVDBENCH_DIR = f'{BASE_DIR}/datasets/aigvdbench/extracted/fake'

# Ablation icin 7 kombinasyon - hangi outputs key'lerinden gelecegi
COMBINATIONS = [
    ('pixel',                 'pixel_prob'),
    ('motion',                'motion_prob'),
    ('consistency',           'consistency_prob'),
    ('pixel+motion',          'pixel_motion_prob'),
    ('pixel+consistency',     'pixel_consistency_prob'),
    ('motion+consistency',    'motion_consistency_prob'),
    ('pixel+motion+consistency (ana)', 'ai_probability'),
]


def compute_metrics(labels, probs, threshold=0.5):
    labels = np.array(labels)
    probs  = np.array(probs)
    preds  = (probs >= threshold).astype(int)

    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    fpr       = fp / (fp + tn + 1e-8)

    return {
        'auc': float(auc), 'f1': float(f1), 'accuracy': float(acc),
        'recall': float(recall), 'precision': float(precision), 'fpr': float(fpr),
        'n': len(labels), 'n_real': int((labels == 0).sum()), 'n_fake': int((labels == 1).sum()),
    }


def load_model(device):
    model = VideoForensicsDetector(freeze_dino=True).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()
    epoch = state.get('epoch', '?')
    auc   = state.get('metrics', {}).get('auc', 0)
    print(f"Checkpoint: epoch={epoch}, val_auc={auc:.4f}\n")
    return model


def get_model_info(model, device):
    """
    Model boyutu, parametre sayisi, checkpoint disk boyutu.
    Bunlar olcum gerektirmez, model nesnesinden ve dosya sisteminden
    dogrudan okunur.
    """
    n_params_total     = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_frozen    = n_params_total - n_params_trainable

    # Bellekteki boyut (float32 varsayimi ile, parametre basina 4 byte)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    # Checkpoint dosyasinin disk boyutu
    checkpoint_size_mb = Path(CHECKPOINT).stat().st_size / 1e6 if Path(CHECKPOINT).exists() else None

    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    return {
        'n_params_total':     int(n_params_total),
        'n_params_trainable': int(n_params_trainable),
        'n_params_frozen':    int(n_params_frozen),
        'model_size_mb':      round(model_size_mb, 2),
        'checkpoint_size_mb': round(checkpoint_size_mb, 2) if checkpoint_size_mb else None,
        'gpu_name':           gpu_name,
    }


def print_model_info(info):
    print(f"{'='*75}")
    print("MODEL BILGISI")
    print(f"{'='*75}")
    print(f"  Toplam parametre:      {info['n_params_total']:,}")
    print(f"  Egitilebilir param.:   {info['n_params_trainable']:,}")
    print(f"  Donmus (frozen) param.: {info['n_params_frozen']:,}")
    print(f"  Model boyutu (RAM):    {info['model_size_mb']:.1f} MB")
    if info['checkpoint_size_mb']:
        print(f"  Checkpoint boyutu (disk): {info['checkpoint_size_mb']:.1f} MB")
    print(f"  GPU:                   {info['gpu_name']}")
    print()


class _TimingTracker:
    """
    Inference suresini ve peak GPU bellegini izler.
    Her batch forward pass'inden once/sonra cagrilir.
    """

    def __init__(self, device):
        self.device = device
        self.total_time = 0.0
        self.total_videos = 0
        self.peak_memory_bytes = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

    def timed_forward(self, model, frames):
        """Forward pass'i zamanlar, sonucu dondurur."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(frames)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        self.total_time += elapsed
        self.total_videos += frames.shape[0]

        if self.device.type == 'cuda':
            current_peak = torch.cuda.max_memory_allocated(self.device)
            self.peak_memory_bytes = max(self.peak_memory_bytes, current_peak)

        return outputs

    def summary(self):
        avg_per_video = self.total_time / max(self.total_videos, 1)
        return {
            'total_inference_time_s':   round(self.total_time, 3),
            'avg_time_per_video_s':     round(avg_per_video, 4),
            'videos_per_second':        round(1.0 / avg_per_video, 2) if avg_per_video > 0 else None,
            'peak_gpu_memory_mb':       round(self.peak_memory_bytes / 1e6, 1) if self.peak_memory_bytes else None,
            'total_videos_processed':   self.total_videos,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'='*75}")
        print("PERFORMANS / KAYNAK KULLANIMI")
        print(f"{'='*75}")
        print(f"  Toplam inference suresi:     {s['total_inference_time_s']:.2f} s")
        print(f"  Video basina ortalama sure:  {s['avg_time_per_video_s']*1000:.1f} ms")
        print(f"  Saniyede islenen video:      {s['videos_per_second']}")
        if s['peak_gpu_memory_mb']:
            print(f"  Peak GPU bellek kullanimi:   {s['peak_gpu_memory_mb']:.1f} MB")
        print(f"  Toplam islenen video:        {s['total_videos_processed']}")
        print()
        return s


class _VideoTimeout(Exception):
    pass


RAW_VIDEO_TIMEOUT = 25  # saniye (sadece bilgi amacli, asagida zorlanmiyor)


def _load_raw_video(path, n_frames=16):
    """
    Ham .mp4 dosyasini dogrudan (subprocess KULLANMADAN) okur.

    NOT - SUBPROCESS DENEMESI BASARISIZ OLDU:
    Daha once SIGALRM (native kodda islenmiyor) ve sonra subprocess+
    terminate() (CUDA baslatilmis bir process'ten spawn edilen worker
    deadlock'a giriyor - bilinen bir PyTorch/CUDA/multiprocessing
    sorunu) denendi, ikisi de bu ortamda (GPU'da model yuklu) calismadi.

    video_io.py'nin kendisi izole test edildiginde (subprocess olmadan,
    dogrudan cagrildiginda) sorunsuz ve hizli calisiyor (~5s/video).
    Bu yuzden burada GERCEK bir OS-seviyesi timeout yok; sadece normal
    Exception handling ile bozuk dosyalar yakalanip atlaniyor. Eger bir
    video gercekten cok uzun surerse (cok nadir, preprocess_final.py
    deneyiminden bilinen oran <%1), script o videoda bekleyebilir -
    ama subprocess'in CUDA ortaminda yarattigi deadlock riskinden
    cok daha guvenli bir secim.
    """
    UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils')
    if UTILS_DIR not in sys.path:
        sys.path.insert(0, UTILS_DIR)
    from video_io import load_video
    bundle = load_video(
        path, n_frames=n_frames, n_semantic=8,
        sampling='window', target_dur=4.0, random_start=False,
        quality_filter=True,
    )
    return bundle.frames_all


# ─────────────────────────────────────────────
# Val kaynagindan degerlendirme
# ─────────────────────────────────────────────

@torch.no_grad()
def run_on_val(model, device, batch_size=8):
    with open(VAL_INDEX) as f:
        samples = list(csv.DictReader(f))

    print(f"Val seti: {len(samples)} ornek\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    tracker = _TimingTracker(device)

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        frames_list = []
        labels = []
        for s in batch:
            try:
                t = torch.load(s['path'], weights_only=True).float()
                frames_list.append(t)
                labels.append(int(s['label']))
            except Exception:
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend(labels)

        if (i // batch_size) % 50 == 0:
            print(f"  {i+len(batch)}/{len(samples)} islendi...")

    perf = tracker.print_summary()
    return all_labels, all_outputs, perf


# ─────────────────────────────────────────────
# AEGIS kaynagindan degerlendirme (keyframe tabanli)
# ─────────────────────────────────────────────

@torch.no_grad()
def run_on_aegis(model, device, batch_size=8):
    import pandas as pd
    from PIL import Image
    import io

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def keyframes_to_tensor(keyframes, n=16, size=224):
        frames = []
        for kf in keyframes[:n]:
            if isinstance(kf, dict) and 'bytes' in kf:
                img = Image.open(io.BytesIO(kf['bytes'])).convert('RGB')
            elif isinstance(kf, Image.Image):
                img = kf.convert('RGB')
            else:
                img = Image.open(io.BytesIO(kf)).convert('RGB')
            img = img.resize((size, size))
            t = torch.from_numpy(np.array(img)).float() / 255.0
            t = t.permute(2, 0, 1)
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            frames.append(t)
        while len(frames) < n:
            frames.append(frames[-1])
        return torch.stack(frames[:n])

    dfs = [pd.read_parquet(f) for f in sorted(Path(AEGIS_DIR).glob('*.parquet'))]
    df = pd.concat(dfs, ignore_index=True)
    print(f"AEGIS Hard Test Set: {len(df)} ornek\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    all_generators = []
    tracker = _TimingTracker(device)

    rows = list(df.iterrows())
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        frames_list = []
        labels = []
        generators = []
        for _, row in batch:
            try:
                meta = json.loads(row['meta_data'])
                label = 1 if meta['ground_truth'] == 'ai' else 0
                t = keyframes_to_tensor(row['keyframes'])
                frames_list.append(t)
                labels.append(label)
                generators.append(meta.get('generator', 'unknown'))
            except Exception:
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend(labels)
        all_generators.extend(generators)

        if (i // batch_size) % 10 == 0:
            print(f"  {i+len(batch)}/{len(rows)} islendi...")

    perf = tracker.print_summary()
    return all_labels, all_outputs, all_generators, perf


# ─────────────────────────────────────────────
# AEGIS kaynagindan degerlendirme (HAM VIDEO - keyframe degil)
# ─────────────────────────────────────────────

@torch.no_grad()
def run_on_aegis_raw(model, device, batch_size=8):
    """
    AEGIS Hard Test Set'i KEYFRAME degil HAM VIDEO (.mp4) olarak okur.

    NEDEN AYRI BIR FONKSIYON:
    run_on_aegis() parquet'teki 8 JPEG keyframe'i kullanir - bu, diger
    modellerin (BusterX, VideoVeritas, Skyra, vs.) tam videoyu kullanarak
    test edilmesinden FARKLI bir girdi formati, adil bir karsilastirma
    saglamiyor. Ozellikle Motion ve Consistency Branch'ler gercek
    temporal bilgiden mahrum kaliyordu (8 keyframe -> 16 frame'e
    tamamlamak icin tekrar ediliyordu).

    Klasor yapisi label/generator bilgisini tasir:
        test_data/real/youtube/*.mp4  -> label=0, generator=camera
        test_data/real/dvf/*.mp4      -> label=0, generator=camera
        test_data/ai_gen/sora/*.mp4   -> label=1, generator=sora
        test_data/ai_gen/kling/*.mp4  -> label=1, generator=kling
    """
    base = Path(AEGIS_RAW_VIDEO_DIR)

    tasks = []  # (path, label, generator)
    for v in (base / 'real' / 'youtube').glob('*.mp4'):
        tasks.append((str(v), 0, 'camera'))
    for v in (base / 'real' / 'dvf').glob('*.mp4'):
        tasks.append((str(v), 0, 'camera'))
    for v in (base / 'ai_gen' / 'sora').glob('*.mp4'):
        tasks.append((str(v), 1, 'sora'))
    for v in (base / 'ai_gen' / 'kling').glob('*.mp4'):
        tasks.append((str(v), 1, 'kling'))

    print(f"AEGIS (ham video): {len(tasks)} dosya bulundu\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    all_generators = []
    n_skipped = 0
    tracker = _TimingTracker(device)

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        frames_list = []
        labels = []
        gens = []
        for path, label, gen in batch:
            try:
                t = _load_raw_video(path)
                frames_list.append(t)
                labels.append(label)
                gens.append(gen)
            except Exception:
                n_skipped += 1
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend(labels)
        all_generators.extend(gens)

        if (i // batch_size) % 10 == 0:
            print(f"  {i+len(batch)}/{len(tasks)} islendi...")

    print(f"  Islendi: {len(all_labels)} | Atlandi (bozuk/kalite): {n_skipped}\n")
    perf = tracker.print_summary()
    return all_labels, all_outputs, all_generators, perf




@torch.no_grad()
def run_on_csv_source(model, device, csv_path, batch_size=8):
    """
    extra_test.csv veya fresh_real_test.csv gibi, ham mp4 path'leri
    iceren herhangi bir CSV'yi degerlendirir.
    """
    with open(csv_path) as f:
        samples = list(csv.DictReader(f))

    print(f"{Path(csv_path).name}: {len(samples)} ornek\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    all_sources = []
    n_skipped = 0
    tracker = _TimingTracker(device)

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        frames_list = []
        labels = []
        srcs = []
        for s in batch:
            try:
                t = _load_raw_video(s['path'])
                frames_list.append(t)
                labels.append(int(s['label']))
                srcs.append(s.get('source', 'unknown'))
            except Exception:
                n_skipped += 1
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend(labels)
        all_sources.extend(srcs)

    print(f"  Islendi: {len(all_labels)} | Atlandi (bozuk/kalite): {n_skipped}\n")
    perf = tracker.print_summary()
    return all_labels, all_outputs, all_sources, perf


# ─────────────────────────────────────────────
# AIGVDBench kapali kaynak modeller (ham mp4, klasor bazli)
# ─────────────────────────────────────────────

@torch.no_grad()
def run_on_aigvdbench(model, device, batch_size=8, n_per_model=50):
    import random
    random.seed(42)

    base = Path(AIGVDBENCH_DIR)
    model_dirs = [d for d in base.iterdir() if d.is_dir()]
    print(f"AIGVDBench kapali kaynak modeller: {[d.name for d in model_dirs]}\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    all_model_names = []
    n_skipped = 0
    tracker = _TimingTracker(device)

    tasks = []
    for model_dir in model_dirs:
        videos = list(model_dir.glob('*.mp4'))
        sample = random.sample(videos, min(n_per_model, len(videos)))
        for v in sample:
            tasks.append((str(v), model_dir.name))

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        frames_list = []
        names = []
        for path, name in batch:
            try:
                t = _load_raw_video(path)
                frames_list.append(t)
                names.append(name)
            except Exception:
                n_skipped += 1
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend([1] * len(frames_list))  # hepsi fake (AIGVDBench/fake/ altinda)
        all_model_names.extend(names)

        if (i // batch_size) % 10 == 0:
            print(f"  {i+len(batch)}/{len(tasks)} islendi...")

    print(f"  Islendi: {len(all_labels)} | Atlandi (bozuk/kalite): {n_skipped}\n")
    perf = tracker.print_summary()
    return all_labels, all_outputs, all_model_names, perf


# ─────────────────────────────────────────────
# GenBuster-Bench++ kaynagindan degerlendirme (ham mp4)
# ─────────────────────────────────────────────

@torch.no_grad()
def run_on_genbuster(model, device, batch_size=8):
    """
    GenBuster-Bench-plusplus/video/{fake,real}/*.mp4 dosyalarini okur.

    Klasor yapisi:
        video/fake/*.mp4  -> label=1 (AI uretimi)
        video/real/*.mp4  -> label=0 (gercek)

    Generator bazinda alt klasor yok, sadece fake/real ayrimi var.
    Bu yuzden generator bazinda breakdown uretilemiyor, sadece
    genel AUC/FPR/Recall/F1 hesaplanabilir.
    """
    base = Path(GENBUSTER_DIR)

    tasks = []
    for v in (base / 'fake').glob('*.mp4'):
        tasks.append((str(v), 1))
    for v in (base / 'real').glob('*.mp4'):
        tasks.append((str(v), 0))

    print(f"GenBuster-Bench++: {len(tasks)} dosya "
          f"(fake={sum(1 for _,l in tasks if l==1)}, "
          f"real={sum(1 for _,l in tasks if l==0)})\n")

    all_outputs = {key: [] for _, key in COMBINATIONS}
    all_labels = []
    n_skipped = 0
    tracker = _TimingTracker(device)

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        frames_list = []
        labels = []
        for path, label in batch:
            try:
                t = _load_raw_video(path)
                frames_list.append(t)
                labels.append(label)
            except Exception:
                n_skipped += 1
                continue
        if not frames_list:
            continue

        frames = torch.stack(frames_list).to(device)
        outputs = tracker.timed_forward(model, frames)

        for _, key in COMBINATIONS:
            all_outputs[key].extend(outputs[key].cpu().numpy().tolist())
        all_labels.extend(labels)

        if (i // batch_size) % 25 == 0:
            print(f"  {i+len(batch)}/{len(tasks)} islendi...")

    print(f"  Islendi: {len(all_labels)} | Atlandi (bozuk/kalite): {n_skipped}\n")
    perf = tracker.print_summary()
    return all_labels, all_outputs, perf


# ─────────────────────────────────────────────
# Ana
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['val', 'aegis', 'aegis_raw', 'extra_test', 'aigvdbench', 'fresh_real', 'genbuster'],
                        default='val')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_per_model', type=int, default=50,
                        help='aigvdbench icin her modelden kac video')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Varsayilan checkpoint yerine kullanilacak .pt dosyasi. '
                             'Ornek: --checkpoint .../epochs/checkpoint_epoch003.pt')
    args = parser.parse_args()

    # --checkpoint verilmisse global sabiti gecici olarak uste yaz
    global CHECKPOINT
    if args.checkpoint:
        CHECKPOINT = args.checkpoint
        print(f"[info] Ozel checkpoint kullaniliyor: {CHECKPOINT}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    model = load_model(device)
    model_info = get_model_info(model, device)
    print_model_info(model_info)

    generators = None
    if args.source == 'val':
        labels, outputs, perf = run_on_val(model, device, args.batch_size)
    elif args.source == 'aegis':
        labels, outputs, generators, perf = run_on_aegis(model, device, args.batch_size)
    elif args.source == 'aegis_raw':
        labels, outputs, generators, perf = run_on_aegis_raw(model, device, args.batch_size)
    elif args.source == 'extra_test':
        labels, outputs, generators, perf = run_on_csv_source(model, device, EXTRA_TEST_CSV, args.batch_size)
    elif args.source == 'fresh_real':
        labels, outputs, generators, perf = run_on_csv_source(model, device, FRESH_REAL_TEST_CSV, args.batch_size)
    elif args.source == 'aigvdbench':
        labels, outputs, generators, perf = run_on_aigvdbench(
            model, device, args.batch_size, args.n_per_model
        )
    elif args.source == 'genbuster':
        labels, outputs, perf = run_on_genbuster(model, device, args.batch_size)
        generators = None  # genbuster'da generator bilgisi yok

    print(f"\n{'='*90}")
    print(f"ABLATION SONUCLARI — Kaynak: {args.source.upper()}")
    print(f"{'='*90}\n")
    print(f"{'Kombinasyon':<35} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Acc':<8} {'FPR':<8}")
    print('-' * 90)

    results = {}
    for name, key in COMBINATIONS:
        m = compute_metrics(labels, outputs[key])
        results[name] = m
        print(f"{name:<35} {m['auc']:<8.4f} {m['precision']:<10.4f} {m['recall']:<8.4f} "
              f"{m['f1']:<8.4f} {m['accuracy']:<8.4f} {m['fpr']:<8.4f}")

    print(f"\nToplam: {len(labels)} ornek | Real: {sum(1 for l in labels if l==0)} | "
          f"Fake: {sum(1 for l in labels if l==1)}")

    # Generator/model bazinda breakdown (AEGIS ve AIGVDBench icin)
    generator_results = {}
    if generators:
        print(f"\n{'='*90}")
        print("GENERATOR/MODEL BAZINDA (ana fusion / ai_probability):")
        print(f"{'='*90}")
        gens = sorted(set(generators))
        for gen in gens:
            idx = [i for i, g in enumerate(generators) if g == gen]
            gen_labels = [labels[i] for i in idx]
            gen_probs  = [outputs['ai_probability'][i] for i in idx]
            m = compute_metrics(gen_labels, gen_probs)
            generator_results[gen] = m
            print(f"  {gen:<15} N={m['n']:<5} Acc={m['accuracy']:.4f} Precision={m['precision']:.4f} "
                  f"Recall={m['recall']:.4f} FPR={m['fpr']:.4f} AUC={m['auc']:.4f}")

    # Kaydet - metrikler + model bilgisi + performans + generator breakdown tek dosyada
    full_results = {
        'source': args.source,
        'model_info': model_info,
        'performance': perf,
        'metrics': results,
        'generator_breakdown': generator_results,
    }
    # Ozel checkpoint kullanildiysa dosya adina epoch bilgisini ekle
    if args.checkpoint:
        epoch_tag = Path(args.checkpoint).stem  # ornek: checkpoint_epoch003
        out_path = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_{args.source}_{epoch_tag}.json')
    else:
        out_path = Path(f'{BASE_DIR}/checkpoints/phase2/ablation_{args.source}.json')
    with open(out_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSonuclar kaydedildi: {out_path}")


if __name__ == '__main__':
    main()
