# 7 Kombinasyon — Threshold Tuning Once/Sonra (Bizim Model)

Her kombinasyon icin ayri ayri: val'de F1-maksimize eden esik bulunur, sonra bu esik o kombinasyonun tum test kaynaklarindaki ham skorlarina uygulanir. Hicbir inference yeniden calistirilmadi.


## AEGIS Hard (ham video) (n=436)

| Kombinasyon | AUC | Tuned Esik | Accuracy once | Accuracy sonra | F1 once | F1 sonra | FPR once | FPR sonra |
|---|---|---|---|---|---|---|---|---|
| Pixel | 0.8606 | 0.0407 | 0.7890 | 0.7775 | 0.7810 | 0.7830 | 0.1743 | 0.2477 |
| Motion | 0.8120 | 0.2177 | 0.6858 | 0.7362 | 0.5836 | 0.7059 | 0.0688 | 0.1606 |
| Consistency | 0.8544 | 0.7850 | 0.7752 | 0.7706 | 0.7870 | 0.7788 | 0.2798 | 0.2661 |
| Pixel+Motion | 0.8911 | 0.1030 | 0.7913 | 0.7913 | 0.7753 | 0.7828 | 0.1376 | 0.1697 |
| Pixel+Consistency | 0.8668 | 0.1410 | 0.7775 | 0.7913 | 0.7749 | 0.7955 | 0.2110 | 0.2294 |
| Motion+Consistency | 0.8787 | 0.7650 | 0.7982 | 0.7959 | 0.8018 | 0.7954 | 0.2202 | 0.2018 |
| Pixel+Motion+Consistency (ana) | 0.8910 | 0.2170 | 0.7936 | 0.7982 | 0.7816 | 0.7925 | 0.1514 | 0.1743 |

## GenBuster-Bench++ (n=2000)

| Kombinasyon | AUC | Tuned Esik | Accuracy once | Accuracy sonra | F1 once | F1 sonra | FPR once | FPR sonra |
|---|---|---|---|---|---|---|---|---|
| Pixel | 0.8725 | 0.0407 | 0.8120 | 0.8065 | 0.7994 | 0.8072 | 0.1250 | 0.1970 |
| Motion | 0.5116 | 0.2177 | 0.5330 | 0.5465 | 0.3366 | 0.4585 | 0.1710 | 0.2910 |
| Consistency | 0.8374 | 0.7850 | 0.7760 | 0.7850 | 0.7846 | 0.7898 | 0.2640 | 0.2380 |
| Pixel+Motion | 0.8609 | 0.1030 | 0.7975 | 0.8070 | 0.7795 | 0.7971 | 0.1210 | 0.1440 |
| Pixel+Consistency | 0.8690 | 0.1410 | 0.8120 | 0.8040 | 0.8052 | 0.8044 | 0.1530 | 0.1980 |
| Motion+Consistency | 0.8461 | 0.7650 | 0.7910 | 0.7940 | 0.7910 | 0.7902 | 0.2090 | 0.1880 |
| Pixel+Motion+Consistency (ana) | 0.8695 | 0.2170 | 0.8005 | 0.8015 | 0.7867 | 0.7918 | 0.1350 | 0.1520 |

## AIGVDBench (n=250, sadece FAKE video — Recall = tespit orani)

| Kombinasyon | Tuned Esik | Recall (once, 0.5) | Recall (sonra) | Delta |
|---|---|---|---|---|
| Pixel | 0.0407 | 0.9120 | 0.9320 | +0.0200 |
| Motion | 0.2177 | 0.4480 | 0.6000 | +0.1520 |
| Consistency | 0.7850 | 0.9320 | 0.9320 | +0.0000 |
| Pixel+Motion | 0.1030 | 0.8960 | 0.9240 | +0.0280 |
| Pixel+Consistency | 0.1410 | 0.9240 | 0.9360 | +0.0120 |
| Motion+Consistency | 0.7650 | 0.9280 | 0.9280 | +0.0000 |
| Pixel+Motion+Consistency (ana) | 0.2170 | 0.9000 | 0.9200 | +0.0200 |

## extra_test (Sora) (n=51, sadece FAKE video — Recall = tespit orani)

| Kombinasyon | Tuned Esik | Recall (once, 0.5) | Recall (sonra) | Delta |
|---|---|---|---|---|
| Pixel | 0.0407 | 0.7059 | 0.7255 | +0.0196 |
| Motion | 0.2177 | 0.2157 | 0.3529 | +0.1373 |
| Consistency | 0.7850 | 0.7647 | 0.7451 | -0.0196 |
| Pixel+Motion | 0.1030 | 0.6667 | 0.6863 | +0.0196 |
| Pixel+Consistency | 0.1410 | 0.6667 | 0.7255 | +0.0588 |
| Motion+Consistency | 0.7650 | 0.7255 | 0.6667 | -0.0588 |
| Pixel+Motion+Consistency (ana) | 0.2170 | 0.6275 | 0.6863 | +0.0588 |
