# AEGIS — AI-Generated Video Detection System
## Proje Özeti

Tamamen yapay zeka tarafından üretilmiş videoları (Sora, Veo, Kling, Runway vb.) gerçek kamera videolarından ayırt eden bir tespit sistemi. **Deepfake/yüz değiştirme değil**, baştan sona sentetik video üretimi tespiti.

---

## Mimari (Faz 2 — Aktif)

```
Video (B, 16, 3, 224, 224)
  ├── Pixel Branch       → DINOv2 (frozen) + LNP + FFT → 512d
  ├── Motion Branch      → LightFlowNet + TemporalTransformer → 512d
  └── Consistency Branch → DINOv2 embedding + velocity + acceleration
                           + speed (||v||) + jerk (||a||) → Transformer → 256d
       ↓
  TripleFusion → ai_probability
  + 6 ek head: 3 tekil (pixel/motion/consistency) + 3 ikili (PM/PC/MC)
  → 7 kombinasyonlu ablation desteği
```

**Eğitilebilir parametre:** 9,231,883 (DINOv2 85M frozen)
**Model boyutu:** 379.8 MB RAM, 453.8 MB disk, peak GPU 2.19 GB

---

## Checkpoint

```
En iyi: $AEGIS_BASE_DIR/checkpoints/phase2/checkpoint_best.pt
        Epoch 7/20, Val AUC=0.9980, Val FPR=0.0218

Tüm epochlar:
        $AEGIS_BASE_DIR/checkpoints/phase2/epochs/checkpoint_epoch00N.pt

Geçmiş: $AEGIS_BASE_DIR/checkpoints/phase2/history.json
```

**Epoch seçim notu:** Val loss kriteri ile epoch 3-4 daha iyi (overfitting epoch 5'te başlıyor), ama Val AUC ve GenBuster'da epoch 7 daha dengeli. Epoch 4 AEGIS AUC'de en yüksek (0.901) ama FPR'de kötü.

---

## Veri Seti (Faz 2)

```
Toplam: 49,370 video → 43,857 train + 4,908 val
Preprocessed: $AEGIS_BASE_DIR/datasets/final_dataset/preprocessed/
  train/index.csv  (43,857 satır)
  val/index.csv    (4,908 satır)

FAKE (~21,870): ModelScope, T2VZ, OpenSora, Pika, VC2, SVD, CogVideo,
                Mora, SEINE, Latte, Veo3, Kling, Gen2, Luma
REAL (~27,500): Kinetics-400 (15K), HD-VG-130M (8K), MSR-VTT/Youku (4.5K)
Format: .pt tensor (16, 3, 224, 224) float16
```

---

## Test Setleri (Eğitime Hiç Girmemiş)

| Kaynak | N | Format | Konum |
|---|---|---|---|
| AEGIS Hard (keyframe) | 436 | parquet | `.../datasets/aegis/data/*.parquet` |
| AEGIS Hard (ham video) | 436 | .mp4 | `.../datasets/aegis_full/videos/test_data/` |
| GenBuster-Bench++ | 2000 | .mp4 | `.../datasets/GenBuster-Bench-plusplus/video/` |
| AIGVDBench | 250 | .mp4 | `.../datasets/aigvdbench/extracted/fake/` |
| extra_test (Sora) | 51 | .mp4 | `.../datasets/final_dataset/extra_test.csv` |
| fresh_real_test | 900 | .mp4 | `.../datasets/final_dataset/fresh_real_test.csv` |

---

## Benchmark Sonuçları — AEGIS Ham Video (n=436)

| Model | AUC | Acc | FPR | Recall | F1 | Süre(s) | Param(B) |
|---|---|---|---|---|---|---|---|
| **Bizim Model** | **0.891** | 0.794 | 0.151 | 0.739 | 0.782 | **0.081** | **0.095** |
| VideoVeritas | 0.882 | 0.878 | 0.161 | 0.922 | 0.887 | 36.67 | 8.77 |
| Skyra | 0.871 | 0.867 | 0.027 | 0.762 | 0.851 | 16.18 | 8.29 |
| IvyFake | 0.778 | 0.626 | 0.711 | 0.972 | 0.724 | 26.21 | 3.75 |
| BusterX | 0.705 | 0.684 | 0.041 | 0.447 | 0.603 | 36.93 | 4.54 |
| CoCoVideo | 0.407 | 0.452 | 0.725 | 0.628 | 0.534 | 0.36 | 0.034 |

**D3 (NSG-VD/XCLIP-16):** AP=0.842 (Real vs Kling), AP=0.780 (Real vs Sora) — farklı metodoloji

---

## Kod Dosyaları

```
src/branches/
  pixel_branch.py        DINOv2 + LNP + FFT, dino_feats çıktıya eklendi
  motion_branch.py       LightFlowNet + TemporalTransformer (değişmedi)
  consistency_branch.py  YENİ — Semantic Stability Branch
  detector_model.py      v2 — TripleFusion + 7 kombinasyon + pos_weight=1.0
  train.py               Faz 2 eğitim scripti (final_dataset CSV'leri)
  ablation_eval.py       7 kombinasyon × 6 kaynak, timing+bellek ölçümü
  build_benchmark_table.py  Tüm modelleri benchmark_table.md/csv'ye yazar
  branch_analysis.py     Disagreement + Correlation + Confidence histogram
  calibration_analysis.py  ECE + Reliability Diagram + Risk-Coverage Curve
  save_scores.py         7 kombinasyonun ham skorlarını .jsonl'e kaydeder
  threshold_tuning.py    Val'de optimal threshold (F1-max) bul, ana kombinasyonu
                          tüm test kaynaklarına uygula (önce/sonra karşılaştırma)
  threshold_tuning_ablation.py  Aynısı ama 7 kombinasyonun HER BİRİ için ayrı ayrı
                          (her kombinasyonun kendi val-optimal eşiği)
  build_full_comparison.py  Bizim model + diğer 5 model, TÜM test setlerinde
                          (AEGIS/GenBuster/AIGVDBench/extra_test), threshold
                          tuning YAPILMADAN (varsayılan eşik) karşılaştırma
  build_fresh_real_test.py  Garantili görülmemiş 900 real video test seti
```

---

## Analiz Çıktıları

```
checkpoints/phase2/
  benchmark_table.md / .csv      AEGIS'te tüm model karşılaştırması (eski, AEGIS-only)
  full_comparison.md / .csv / .png   Bizim model + 5 model, TÜM test setlerinde
                                  (AEGIS/GenBuster/AIGVDBench/extra_test),
                                  threshold tuning YAPILMADAN — build_full_comparison.py çıktısı
  ablation_*.json                Her kaynak × her kombinasyon metrikleri (varsayılan eşik)
  history.json                   20 epoch eğitim geçmişi
  scores/                        Ham skorlar (.jsonl) — inference gerektirmez
    val_scores.jsonl
    aegis_raw_scores.jsonl
    genbuster_scores.jsonl
    aigvdbench_scores.jsonl
    extra_test_scores.jsonl
  branch_analysis/               Disagreement + korelasyon grafikleri
  threshold_tuning.json          Ana kombinasyon: val-optimal eşik analizi (F1-max, Youden-J)
  threshold_comparison.json/.md  Ana kombinasyon: önce(0.5)/sonra(tuned) tüm kaynaklarda
  threshold_tuning_ablation.json 7 kombinasyonun HER BİRİ için val-optimal eşik
  threshold_comparison_ablation.json/.md/.png
                                  7 kombinasyon × 4 kaynak, önce/sonra karşılaştırma (görsel)
```

---

## Yapılacaklar (Sıradaki Adımlar)

### Tamamlandı
- [x] `save_scores.py --source all` çalıştırıldı (ham skorlar `scores/*.jsonl`'de, bir daha inference gerekmez)
- [x] `threshold_tuning.py` + `threshold_tuning_ablation.py` çalıştırıldı — val-optimal eşik (F1-max)
      hem ana kombinasyon hem 7 kombinasyonun her biri için ayrı ayrı bulundu, tüm test
      kaynaklarına (inference'sız) uygulanıp önce/sonra karşılaştırıldı. Sonuç: genelleşiyor
      ama bedelli (recall/F1 hafif artıyor, FPR de artıyor) — bkz. Önemli Kararlar.
- [x] `build_full_comparison.py` ile bizim model + 5 model TÜM test setlerinde karşılaştırıldı
      (daha önce sadece AEGIS'te vardı)

### Acil (Şu An)
- [ ] `calibration_analysis.py --source aegis_raw` çalıştır (risk-coverage hatası düzeltildi)



---

## Önemli Kararlar ve Gerekçeler

**pos_weight=1.0 (nötr):** Veri seti dengeli (~1:1.26), pos_weight ile FPR'yi doğrudan düşürmeye çalışmak amaç fonksiyonunu değerlendirme metriğiyle karıştırır.

**Checkpoint epoch 7:** Val AUC'ye göre seçildi. Epoch 4 AEGIS'te daha yüksek AUC (0.901) ama FPR kötü (0.206). Epoch 7 GenBuster'da daha dengeli (AUC 0.870, FPR 0.135).

**WaveRep/Wavelet:** Augmentasyon Faz 3'e ertelendi (paired data gerektirir). Sadece mimari değişiklik (FFT→Wavelet) ayrı izole deney olarak yapılacak.

**Consistency Branch:** Deneysel, ablation ile doğrulandı. AEGIS'te Motion'dan tutarlı şekilde daha güçlü. Pixel↔Consistency korelasyonu 0.86 (yüksek) — makale için dürüst bir sınırlama notu gerekiyor.

**Motion Branch zayıflığı:** Tüm zor test setlerinde sistematik (AEGIS recall 0.44, GenBuster recall 0.24). Optik akış tabanlı yaklaşım yeni nesil modellere karşı kör. Faz 3'te yeniden tasarım planlanıyor.

**Threshold tuning (varsayılan 0.5 → val-optimal 0.217, ana kombinasyon):** Val'de neredeyse etkisiz (F1 +0.001, zaten AUC 0.998 ile çok iyi ayrışıyor). Zor test setlerine (AEGIS/GenBuster) taşındığında recall/F1 hafif iyileşiyor ama FPR de aynı oranda kötüleşiyor (AEGIS +2.3 puan, GenBuster +1.7 puan) — net kazanç değil, bir trade-off. `pos_weight=1.0` kararındaki gerekçeyle tutarlı: FPR'yi düşürmeye çalışmak (veya threshold ile telafi etmek) amaç fonksiyonunu karıştırıyor. **Motion branch tek başına** tuned eşikle belirgin iyileşiyor (AEGIS F1 0.58→0.71, AIGVDBench recall 0.45→0.60) çünkü 0.5 eşiği motion için kalibrasyonsuz (val-optimal eşiği 0.5 değil 0.218) — ama GenBuster'da AUC=0.51 (rastgele düzeyinde), yani eşik ayarı sıralama gücünü değiştiremiyor. **Consistency ve Motion+Consistency** ise tam tersi: val-optimal eşikleri ~0.78-0.79 (0.5'in çok üstü), extra_test'te tuning sonrası recall düşüyor (-0.02, -0.06) — tuning her koşulda iyileştirmiyor. Checkpoint epoch 7 + threshold 0.5 kombinasyonu muhtemelen bilinçli bir denge noktası, değiştirilmedi. Detay: `threshold_comparison_ablation.png`.

---

## Diğer Modeller (Karşılaştırma İçin)

```
$AEGIS_BASE_DIR/ai_video_detector/other_models/
  CoCoVideo/      VideoVeritas/    Skyra/
  IvyFake/        BusterX/        D3/
  compute_metrics.py      Sonuç CSV/JSON'ları manifest ile birleştirip metrik hesaplar.
                          --fakeonly modu (AIGVDBench/extra_test için) düzeltildi —
                          eskiden final tablo adımında KeyError ile çöküyordu.
  comparison_report.csv   AEGIS sonuçları (real+fake, n=436)
  report_genbuster.csv    GenBuster-Bench++ sonuçları (real+fake, n=2000)
  report_aigvdbench.csv   AIGVDBench sonuçları (fake-only, n=250, sadece Recall)
  report_extratest.csv    extra_test/Sora sonuçları (fake-only, n=51, sadece Recall)
```

Her model için `infer.py` scripti var, aynı test videolarını kullanıyor.

**Not — retry dosyaları:** VideoVeritas ve BusterX'in bazı AIGVDBench/extra_test videolarında
ilk çalıştırmada `ERROR` dönmüştü, ayrı `*_retry.csv` dosyalarında düzeltilmiş. `report_*.csv`
üretilirken bunlar video-path bazında merge edilerek (`results_*_merged.csv`) kullanıldı —
sadece retry dosyasını kullanmak eksik/yanlış sonuç veriyordu (örn. VideoVeritas AIGVDBench
recall'ü retry-only ile yanlışlıkla 0.50 çıkmıştı, merge sonrası doğrusu 0.964).

---

## Bağlam: Paralel Chat

Bu projenin **6 açık kaynak model karşılaştırması** ayrı bir Claude chat'te yürütüldü. Sonuçlar `comparison_report.csv`'de birleştirildi ve `build_benchmark_table.py` ile ana tabloya eklendi. İleride yeni modeller eklenince aynı scripti çalıştırmak yeterli.
