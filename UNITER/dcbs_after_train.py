"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

run inference for Image Text Retrieval
"""
import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from apex import amp
from horovod import torch as hvd

from data import (PrefetchLoader, ImageLmdbGroup,
                  DetectFeatLmdb, TxtTokLmdb, OodDataset, ood_collate,
                  OodVcrDataset, VcrTxtTokLmdb)
from model.pretrain import UniterForPretraining
from model.pretrain_vcr import UniterForPretrainingForVCR

from utils.logger import LOGGER
from utils.misc import Struct, parse_with_config
from utils.const import IMG_DIM, IMG_LABEL_DIM, BUCKET_SIZE


def load_img_feat(db_list, all_img_dbs, opts):
    db_ = db_list.split(";")
    assert len(db_) <= 2, "More than two img_dbs found"
    gt_db_path, db_path = "", ""
    for d in db_:
        if "gt" in d:
            gt_db_path = d
        else:
            db_path = d
    if gt_db_path != "":
        img_db_gt = DetectFeatLmdb(
            gt_db_path, -1, opts.max_bb, opts.min_bb, 100,
            opts.compressed_db)
        all_img_dbs.path2imgdb[gt_db_path] = img_db_gt
    else:
        img_db_gt = None
    img_db = all_img_dbs[db_path] if db_path != "" else None
    all_img_dbs.path2imgdb[db_path] = img_db
    return img_db, img_db_gt


def build_dataloader(dataset, collate_fn, opts):
    batch_size = opts.val_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=opts.n_workers, shuffle=False,
                                pin_memory=opts.pin_mem, collate_fn=collate_fn)
    dataloader = PrefetchLoader(dataloader)
    return dataloader


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    # Prepare model
    model = UniterForPretrainingForVCR.from_pretrained(
            opts.model_config, {},
            img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
    model.init_type_embedding()
    model.init_word_embedding(81)

    checkpoint = torch.load(opts.checkpoint)
    state_dict = checkpoint.get('model_state', checkpoint)
    matched_state_dict = {}
    unexpected_keys = set()
    missing_keys = set()
    for name, param in model.named_parameters():
        missing_keys.add(name)
    for key, data in state_dict.items():
        if key in missing_keys:
            matched_state_dict[key] = data
            missing_keys.remove(key)
        else:
            unexpected_keys.add(key)
    print("Unexpected_keys:", list(unexpected_keys))
    print("Missing_keys:", list(missing_keys))
    model.load_state_dict(matched_state_dict, strict=False)
    del checkpoint

    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    val_txt_db = VcrTxtTokLmdb(opts.val_txt_db, -1)
    val_img_db, val_img_db_gt = load_img_feat(
        opts.val_img_db, all_img_dbs, opts)
    val_dataset = OodVcrDataset("test", val_txt_db, img_db=val_img_db, img_db_gt=val_img_db_gt)
    val_dataloader = build_dataloader(val_dataset, ood_collate, opts)
    evaluate(model, val_dataloader, opts)


@torch.no_grad()
def evaluate(model, val_loader, opts):
    val_pbar = tqdm(total=len(val_loader))
    model.eval()
    pooled_reps = []
    img_fnames = []
    for i, batch in enumerate(val_loader):
        pooled_rep = model(batch, task='ood', compute_loss=opts.enable_con_head)
        pooled_reps.append(pooled_rep)
        img_fnames.extend(batch['img_fname'])
        val_pbar.update(1)
    torch.save({"pooled_rep": torch.cat(pooled_reps, dim=0), "img_fname": img_fnames}, os.path.join(opts.output_dir, "pooled_reps.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db", default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--img_db", default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="model checkpoint binary")
    parser.add_argument("--model_config", default=None, type=str,
                        help="model config json")
    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the inference results will be "
             "written.")

    # optional parameters
    parser.add_argument("--train_config", default=None, type=str,
                        help="hps.json from training (for prepro hps)")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')
    parser.add_argument("--batch_size", default=400, type=int,
                        help="number of tokens in a batch")

    # device parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--config', required=True, help='JSON config files')


    args = parse_with_config(parser)

    main(args)
