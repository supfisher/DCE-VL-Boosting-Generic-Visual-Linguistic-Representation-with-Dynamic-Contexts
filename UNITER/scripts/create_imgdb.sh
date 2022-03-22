# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

IMG_NPY=$1
OUT_DIR=$2

set -e

echo "converting image features ..."
if [ ! -d $OUT_DIR ]; then
    mkdir -p $OUT_DIR
fi
NAME=$(basename $IMG_NPY)
docker run --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUT_DIR,dst=/img_db,type=bind \
    --mount src=$IMG_NPY,dst=/$NAME,type=bind,readonly \
    -w /src chenrocks/uniter \
    python scripts/convert_imgdir.py --img_dir /$NAME --output /img_db
    python -m pdb scripts/convert_visualcomet_imgdb.py --img_dir /ibex/scratch/mag0a/Github/visual-comet/data/features --split 'train' --output dataset/visualcomet/img_db/train
echo "done"
