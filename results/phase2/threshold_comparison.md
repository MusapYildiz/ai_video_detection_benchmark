# Threshold Tuning — Once/Sonra Karsilastirmasi (Bizim Model)

Val setinde F1-maksimize eden esik: **0.2170** (varsayilan: 0.5)

Not: Hicbir inference yeniden calistirilmadi, sadece kayitli ham olasiliklara (ai_probability) farkli karar esigi uygulandi.


## Val (n=4908) — esik secildigi kume

| Esik | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Varsayilan (0.5) | 0.9770 | 0.9723 | 0.9755 | 0.9739 | 0.0218 |
| Tuned (F1-opt) | 0.9778 | 0.9702 | 0.9796 | 0.9749 | 0.0236 |

## AEGIS Hard (ham video) (n=436, AUC=0.8910)

| Esik | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Once (0.5) | 0.7936 | 0.8299 | 0.7385 | 0.7816 | 0.1514 |
| Sonra (0.2170) | 0.7982 | 0.8155 | 0.7706 | 0.7925 | 0.1743 |

Delta: accuracy +0.0046, F1 +0.0109, FPR +0.0229


## GenBuster-Bench++ (n=2000, AUC=0.8695)

| Esik | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Once (0.5) | 0.8005 | 0.8450 | 0.7360 | 0.7867 | 0.1350 |
| Sonra (0.2170) | 0.8015 | 0.8324 | 0.7550 | 0.7918 | 0.1520 |

Delta: accuracy +0.0010, F1 +0.0051, FPR +0.0170


## AIGVDBench (n=250, sadece FAKE — Recall = tespit orani)

| Esik | Recall |
|---|---|
| Once (0.5) | 0.9000 |
| Sonra (0.2170) | 0.9200 |

Delta: recall +0.0200


## extra_test (Sora) (n=51, sadece FAKE — Recall = tespit orani)

| Esik | Recall |
|---|---|
| Once (0.5) | 0.6275 |
| Sonra (0.2170) | 0.6863 |

Delta: recall +0.0588

