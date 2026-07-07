#!/bin/bash
# extract_dataset.sh — GenVideo-100K arsivlerini ac

DATASET_DIR="${AEGIS_BASE_DIR:-/kullanici_yedek/musap.yildiz}/datasets/genvideo/cccnju/GenVideo-100K"
EXTRACT_DIR="${AEGIS_BASE_DIR:-/kullanici_yedek/musap.yildiz}/datasets/genvideo/extracted"

mkdir -p "$EXTRACT_DIR/fake"
mkdir -p "$EXTRACT_DIR/real"

echo "Extract basliyor..."
echo "Kaynak: $DATASET_DIR"
echo "Hedef:  $EXTRACT_DIR"
echo ""

# Fake videolar — .tar.gz dosyalari
for file in Latte OpenSora pika SEINE SVD ZeroScope; do
    archive="$DATASET_DIR/${file}.tar.gz"
    target="$EXTRACT_DIR/fake/$file"

    if [ -d "$target" ]; then
        echo "[ATLA] $file zaten extract edilmis"
        continue
    fi

    if [ ! -f "$archive" ]; then
        echo "[YOK]  $archive bulunamadi"
        continue
    fi

    echo "[...] $file extract ediliyor ($(du -h $archive | cut -f1))..."
    mkdir -p "$target"
    tar xzf "$archive" -C "$target" --strip-components=1 2>/dev/null || \
    tar xzf "$archive" -C "$target" 2>/dev/null
    echo "[OK]  $file tamamlandi → $(find $target -name '*.mp4' | wc -l) video"
done

# Real videolar — uzantisiz gzip
echo ""
echo "[...] Real_part_aa extract ediliyor (30GB, uzun surebilir)..."
real_target="$EXTRACT_DIR/real/msrvtt_youku"
mkdir -p "$real_target"

tar xzf "$DATASET_DIR/Real_part_aa" -C "$real_target" --strip-components=1 2>/dev/null || \
tar xzf "$DATASET_DIR/Real_part_aa" -C "$real_target" 2>/dev/null
echo "[OK]  Real_part_aa tamamlandi → $(find $real_target -name '*.mp4' | wc -l) video"

# Ozet
echo ""
echo "===== EXTRACT OZETI ====="
for dir in "$EXTRACT_DIR/fake"/*/; do
    n=$(find "$dir" -name "*.mp4" 2>/dev/null | wc -l)
    echo "  fake/$(basename $dir): $n video"
done
for dir in "$EXTRACT_DIR/real"/*/; do
    n=$(find "$dir" -name "*.mp4" 2>/dev/null | wc -l)
    echo "  real/$(basename $dir): $n video"
done

echo ""
echo "Toplam fake: $(find $EXTRACT_DIR/fake -name '*.mp4' | wc -l)"
echo "Toplam real: $(find $EXTRACT_DIR/real -name '*.mp4' | wc -l)"
echo ""
echo "Egitim komutu:"
echo "  CUDA_VISIBLE_DEVICES=4 python3 train.py \\"
echo "    --data_root $EXTRACT_DIR \\"
echo "    --max_per_class 15000 \\"
echo "    --epochs 20 \\"
echo "    --batch_size 8 \\"
echo "    --num_workers 4"
