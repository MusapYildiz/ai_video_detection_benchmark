"""
CoCoVideo (CoCoDetect) - Inference-only script
------------------------------------------------
Repo'da hazır bir inference scripti olmadığı için bu script train.py / utils/paired_model.py /
utils/paired_dataset.py içeriğine bakılarak, checkpoint'in args'ına (T=32, frame_size=224,
backbone=r3d_18, emb_dim=128) tam uyumlu şekilde yazılmıştır.

Label convention (orijinal repo ile aynı):
    confidence > 0.5  ->  REAL
    confidence <= 0.5 ->  FAKE

Kullanım:
    # Tek video dosyası (mp4/avi/...) üzerinde:
    python infer.py --checkpoint checkpoint.pth --video /path/to/video.mp4

    # Önceden frame'lere ayrılmış bir klasör üzerinde (0000.jpg, 0001.jpg, ... sıralı):
    python infer.py --checkpoint checkpoint.pth --frames_dir /path/to/frames_folder

    # Bir klasördeki TÜM videoları toplu işlemek için:
    python infer.py --checkpoint checkpoint.pth --video_dir /path/to/videos/ --output results.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from model_stats_utils import get_param_count


# ----------------------------------------------------------------------------
# Model tanımı - utils/paired_model.py ile BİREBİR aynı (repo'ya dokunmadan,
# kendi başına çalışabilmesi için buraya kopyalandı)
# ----------------------------------------------------------------------------
class PairedContrastiveModel(nn.Module):
    def __init__(self, backbone_name='r3d_18', emb_dim=128, pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == 'r3d_18':
            self.backbone = torchvision.models.video.r3d_18(weights=None)
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone_type = 'r3d'
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone_name}")

        self.confidence_head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, emb_dim), nn.BatchNorm1d(emb_dim)
        )

    def forward_single(self, x):
        # x: [B, C, T, H, W]
        feat = self.backbone(x)
        confidence = self.confidence_head(feat)
        return confidence


# ----------------------------------------------------------------------------
# Frame okuma / örnekleme - utils/paired_dataset.py ile aynı mantık:
#   num_frames <= T  -> son frame'i tekrarlayarak T'ye tamamla
#   num_frames >  T  -> yalnızca İLK T frame alınır (eğitimdeki davranışla aynı)
# ----------------------------------------------------------------------------
R3D_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
R3D_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def sample_indices(num_frames, T):
    if num_frames <= T:
        return list(range(num_frames)) + [num_frames - 1] * (T - num_frames)
    return list(range(T))


def load_frames_from_dir(frames_dir, T, frame_size):
    files = sorted([f for f in os.listdir(frames_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if len(files) == 0:
        raise ValueError(f"Frame bulunamadı: {frames_dir}")

    indices = sample_indices(len(files), T)
    frames = []
    for idx in indices:
        img = cv2.imread(os.path.join(frames_dir, files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (frame_size, frame_size):
            img = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
    return frames


def load_frames_from_video(video_path, T, frame_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video açılamadı: {video_path}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (frame_size, frame_size):
            frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
        all_frames.append(frame)
    cap.release()

    if len(all_frames) == 0:
        raise ValueError(f"Videodan hiç frame okunamadı: {video_path}")

    indices = sample_indices(len(all_frames), T)
    return [all_frames[i] for i in indices]


def frames_to_tensor(frames):
    # frames: list of [H, W, 3] uint8 RGB
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # [T, H, W, 3]
    arr = (arr - R3D_MEAN) / R3D_STD
    arr = arr.transpose(3, 0, 1, 2)                              # [3, T, H, W]
    tensor = torch.from_numpy(arr).float().unsqueeze(0)          # [1, 3, T, H, W]
    return tensor


def predict_one(model, device, T, frame_size, video_path=None, frames_dir=None):
    if frames_dir is not None:
        frames = load_frames_from_dir(frames_dir, T, frame_size)
    else:
        frames = load_frames_from_video(video_path, T, frame_size)

    x = frames_to_tensor(frames).to(device)
    with torch.no_grad():
        confidence = model.forward_single(x)
    score = confidence.item()
    label = "REAL" if score > 0.5 else "FAKE"
    return score, label


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt['args']

    model = PairedContrastiveModel(
        backbone_name=train_args['backbone'],
        emb_dim=train_args['emb_dim'],
        pretrained=False,   # zaten checkpoint'ten ağırlık yükleyeceğiz
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    T = train_args['num_frames']
    frame_size = train_args['frame_size']
    print(f"[info] Checkpoint epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.2f}%  "
          f"T={T}  frame_size={frame_size}  backbone={train_args['backbone']}")
    return model, T, frame_size


def main():
    parser = argparse.ArgumentParser(description="CoCoVideo (CoCoDetect) inference")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--video', type=str, default=None, help="Tek video dosyası")
    parser.add_argument('--frames_dir', type=str, default=None, help="Önceden çıkarılmış frame klasörü")
    parser.add_argument('--video_dir', type=str, default=None,
                         help="İçinde birden çok video olan klasör (toplu çıkarım)")
    parser.add_argument('--output', type=str, default='results.csv', help="--video_dir kullanılırsa çıktı CSV yolu")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if sum(x is not None for x in [args.video, args.frames_dir, args.video_dir]) != 1:
        sys.exit("Tam olarak bir tanesini ver: --video, --frames_dir veya --video_dir")

    device = torch.device(args.device)
    model, T, frame_size = load_model(args.checkpoint, device)

    param_count = get_param_count(model)
    # NOT: get_dir_size_gb bir KLASOR bekliyor; CoCoVideo'da tek bir .pth dosyasi var,
    # o yuzden checkpoint dosyasinin kendi boyutunu kullaniyoruz.
    checkpoint_path = Path(args.checkpoint)
    disk_size_gb = checkpoint_path.stat().st_size / (1024 ** 3) if checkpoint_path.exists() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.video_dir is not None:
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        video_paths = sorted(
            p for p in Path(args.video_dir).rglob('*')
            if p.suffix.lower() in video_exts
        )
        print(f"[info] {len(video_paths)} video bulundu.")

        rows = []
        for vp in video_paths:
            t0 = time.perf_counter()
            try:
                score, label = predict_one(model, device, T, frame_size, video_path=vp)
                elapsed = time.perf_counter() - t0
                rows.append([str(vp), score, label, f"{elapsed:.2f}"])
                print(f"{vp.name:40s}  score={score:.4f}  -> {label}  ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                rows.append([str(vp), "ERROR", str(e), f"{elapsed:.2f}"])
                print(f"{vp.name:40s}  HATA: {e}  ({elapsed:.2f}s)")

        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'confidence_score', 'prediction', 'inference_seconds'])
            writer.writerows(rows)
        print(f"\n[info] Sonuçlar kaydedildi: {args.output}")

        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None
        model_info = {
            'model_name': 'CoCoVideo',
            'param_count': param_count,
            'param_count_billions': round(param_count / 1e9, 4),
            'disk_size_gb': round(disk_size_gb, 4) if disk_size_gb is not None else None,
            'peak_gpu_memory_gb': round(peak_gb, 2) if peak_gb is not None else None,
        }
        info_path = Path(args.output).with_name('model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2)
        print(f"[info] Model bilgisi kaydedildi: {info_path}")
        print(f"[info] {model_info}")

    else:
        score, label = predict_one(
            model, device, T, frame_size,
            video_path=args.video, frames_dir=args.frames_dir
        )
        print(f"\nSkor (confidence): {score:.4f}")
        print(f"Tahmin: {label}   (confidence > 0.5 -> REAL, <= 0.5 -> FAKE)")


if __name__ == '__main__':
    main()
