"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb
import pickle
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
VCR_FEATURES_DIR = '/ibex/scratch/mag0a/Github/visual-comet/data/features'
VCR_IMAGES_DIR = '/ibex/scratch/mag0a/Github/visual-comet/data/vcr1images'# os.environ['VCR_PARENT_DIR']
record_cache = '/ibex/scratch/mag0a/Github/VL-BERT/pretrain/data/visualcomet_preparetion/cache/event_similarity_cleaned/'


with open(os.path.join(os.path.dirname(__file__), 'cocoontology.json'), 'r') as f:
    coco = json.load(f)
coco_objects = ['__background__'] + [x['name'] for k, x in sorted(coco.items(), key=lambda x: int(x[0]))]
coco_obj_to_ind = {o: i for i, o in enumerate(coco_objects)}


def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, len(img_dump['object_features']))
    num_bb = min(max_bb, num_bb)
    return int(num_bb)


@curry
def load_pkl(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    with open(os.path.join(VCR_FEATURES_DIR, fname), 'rb') as f:
        img_dump = pickle.load(f)

    name = '.'.join(basename(fname).split('.')[:-1]) + '.jpg'
    assert name in img2metadata, "not recorded images..."
    with open(os.path.join(VCR_IMAGES_DIR, img2metadata[name]), 'r') as f:
        metadata = json.load(f)
    boxes = np.array(metadata['boxes'])
    w, h = metadata['width'], metadata['height']
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / h
    # img_dump['norm_bb'] = np.append(boxes[:, :4], np.array([[w/1200, h/600] for _ in range(len(boxes))]), axis=1)
    img_dump['norm_bb'] = np.append(boxes[:, :4], boxes[:, 2:4] - boxes[:, 0:2], axis=1)
    img_dump['soft_labels'] = np.array([coco_obj_to_ind[_] for _ in metadata['names']])

    if keep_all:
        nbb = None
    else:
        nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
    dump = {}
    for key, arr in img_dump.items():
        if key == 'object_features':
            key = 'features'
        if arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        if arr.ndim == 2:
            dump[key] = arr[:nbb, :]
        elif arr.ndim == 1:
            dump[key] = arr[:nbb]

    return name, dump, nbb


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    if opts.img_dir[-1] == '/':
        opts.img_dir = opts.img_dir[:-1]
    split = basename(opts.img_dir)
    if opts.keep_all:
        db_name = 'all'
    else:
        if opts.conf_th == -1:
            db_name = f'feat_numbb{opts.num_bb}'
        else:
            db_name = (f'feat_th{opts.conf_th}_max{opts.max_bb}'
                       f'_min{opts.min_bb}')
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{split}'):
        os.makedirs(f'{opts.output}/{split}')
    env = lmdb.open(f'{opts.output}/{split}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    # files = glob.glob(f'{opts.img_dir}/*.pkl')
    files = opts.files
    load = load_pkl(opts.conf_th, opts.max_bb, opts.min_bb, opts.num_bb,
                    keep_all=opts.keep_all)
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    if opts.conf_th != -1 and not opts.keep_all:
        with open(f'{opts.output}/{split}/'
                  f'nbb_th{opts.conf_th}_'
                  f'max{opts.max_bb}_min{opts.min_bb}.json', 'w') as f:
            json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str,
                        help="The input images.")
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument("--output", default=None, type=str,
                        help="output lmdb")
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--keep_all', action='store_true',
                        help='keep all features, overrides all following args')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    args = parser.parse_args()

    feature_files = []
    img2metadata = {}

    image_clusters = []
    path = "/ibex/scratch/mag0a/Github/visual-comet/data/vcr1images"
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            # if 'lsmdc' in dir_name:  # only lsmdc dir has time stamp
            files = [os.path.join(dir_name, f) for f in os.listdir(os.path.join(path, dir_name)) if ('jpg' in f) and ('npy' not in f)]
            files = sorted(files)
            for img_fn in files:
                img_fn = '.'.join(basename(img_fn).split('.')[:-1])
                feature_files.append(img_fn + '.pkl')
                img2metadata[img_fn + '.jpg'] = os.path.join(dir_name, img_fn) + '.json'

    args.files = feature_files
    main(args)
