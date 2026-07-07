# AEGIS Hard Test Set — Benchmark Karsilastirma Tablosu

Toplam ornek: 436 (218 real + 111 kling + 107 sora)

## Genel Performans

| Model | Accuracy | Precision | Recall | F1 | AUC | Sure(s) | Param(B) | Disk(GB) | GPU(GB) |
|---|---|---|---|---|---|---|---|---|---|
| Bizim Model (ham video) | 0.7936 | 0.8299 | 0.7385 | 0.7816 | 0.8910 | 0.081 | 0.095 | 0.44 | 2.14 |
| VideoVeritas | 0.8784 | 0.8547 | 0.9217 | 0.8869 | 0.8818 | 36.670 | 8.770 | 16.34 | 30.17 |
| Skyra | 0.8670 | 0.9651 | 0.7615 | 0.8513 | 0.8706 | 16.180 | 8.290 | 15.46 | 20.62 |
| IvyFake | 0.6261 | 0.5769 | 0.9722 | 0.7241 | 0.7776 | 26.210 | 3.750 | 7.03 | 12.12 |
| BusterX | 0.6835 | 0.9271 | 0.4472 | 0.6034 | 0.7051 | 36.930 | 4.540 | 9.66 | 14.08 |
| CoCoVideo | 0.4518 | 0.4644 | 0.6284 | 0.5341 | 0.4069 | 0.360 | 0.034 | 0.37 | 0.63 |

## Generator Bazinda Accuracy

| Model | Camera (Real) | Kling | Sora |
|---|---|---|---|
| Bizim Model (ham video) | 0.8486 | 0.7117 | 0.7664 |
| VideoVeritas | 0.8394 | 0.9550 | 0.8785 |
| Skyra | 0.9725 | 1.0000 | 0.5140 |
| IvyFake | 0.2890 | 0.9820 | 0.9439 |
| BusterX | 0.9587 | 0.4414 | 0.3738 |
| CoCoVideo | 0.2752 | 0.5045 | 0.7570 |

## Ek Test Kaynaklari (Bizim Model — Sadece Fake Video)

Not: Bu kaynaklar yalnizca AI uretimi video iceriyor, FPR hesaplanamaz. Anlamli metrik Recall'dur.

| Kaynak | N | Accuracy | Recall | F1 | Sure(s) |
|---|---|---|---|---|---|
| extra_test (Sora) | 51 | 0.6275 | 0.6275 | 0.7711 | 0.089 |
| aigvdbench | 250 | 0.9000 | 0.9000 | 0.9474 | 0.084 |

### Generator Bazinda Accuracy (Ek Kaynaklar)

| Kaynak | Generator | Accuracy |
|---|---|---|

## D3 (NSG-VD / XCLIP-16) — AP Skorlari

Not: Farkli metodoloji (Average Precision, real vs tek generator ikili karsilastirma). Diger modellerle dogrudan ayni sutunlarda kiyaslanamaz.

| Real vs Generator | AP Score |
|---|---|
| all_fake | 0.6935 |
| kling | 0.8424 |
| sora | 0.7800 |
