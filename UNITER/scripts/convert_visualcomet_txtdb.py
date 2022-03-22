"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import pickle
import os
from os.path import exists
from pytorch_pretrained_bert import BertTokenizer
from cytoolz import curry
from tqdm import tqdm

from data.data import open_lmdb
import sys
sys.path.append("/ibex/scratch/mag0a/Github/visual-comet/")
from utils.file_utils import read_and_parse_finetune_json

record_cache = '/ibex/scratch/mag0a/Github/VL-BERT/pretrain/data/visualcomet_preparetion/cache/event_similarity_cleaned/'
vcg_dir='/ibex/scratch/mag0a/Github/visual-comet/data/visualcomet_annotations/'


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


def process_visualcomet(jsonl, db, tokenizer, missing=None):
    id2len = {}
    txt2img = {}
    lines = jsonl.readlines()
    for id_, l in enumerate(lines):
        img_fn, metadata_fn, caption = l.split(",")
        input_ids = tokenizer('[CLS] '+caption+' [SEP]')
        img_fname = img_fn.split('/')[-1]
        txt2img[str(id_)] = img_fname
        id2len[str(id_)] = len(input_ids)
        db[str(id_)] = {'input_ids': input_ids, 'img_fname': img_fname}
    return id2len, txt2img


def load_records(split):
    split_filename = '{}_annots.json'.format(split)
    vcg_path = os.path.join(vcg_dir, split_filename)
    print("Loading vcg dataset {}".format(vcg_path))

    records = read_and_parse_finetune_json(vcg_path)
    # with open(vcg_path, 'rb') as f:
    #     records = json.load(f)
    return records


def process_eoc(records, db, tokenizer):
    id2len = {}
    txt2img = {}
    event_label = {'event': 1, 'intent': 1, 'before': 2, 'after': 3}
    for id_, record in enumerate(records):
        img_fn, metadata_fn = record['img_fn'], record['metadata_fn']
        relationship_labels = event_label[record['inference_relation']]
        if record['inference_relation'] in ['event', 'intent']:
            caption = record['event']
        else:
            caption = record['inference_text']
        input_ids = tokenizer('[CLS] ' + caption + ' [SEP]')
        img_fname = img_fn.split('/')[-1]
        txt2img[str(id_)] = img_fname
        id2len[str(id_)] = len(input_ids)
        db[str(id_)] = {'input_ids': input_ids, 'img_fname': img_fname, 'label': relationship_labels}
    return id2len, txt2img


def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.bert
    toker = BertTokenizer.from_pretrained(
        opts.bert, do_lower_case='uncased' in opts.bert)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    output_field_name = ['id2len', 'txt2img']

    if opts.task == 'visualcomet':
        parse_record_cache = os.path.join(record_cache, '{}_record_img_event.txt'.format(opts.split))
        with open_db() as db:
            with open(parse_record_cache, 'r+', encoding="utf-8") as ann:

                if opts.missing_imgs is not None:
                    missing_imgs = set(json.load(open(opts.missing_imgs)))
                else:
                    missing_imgs = None
                jsons = process_visualcomet(
                    ann, db, tokenizer, missing_imgs)
    elif opts.task == 'eoc':
        records = load_records(opts.split)
        with open_db() as db:
            jsons = process_eoc(
                records, db, tokenizer)

    for dump, name in zip(jsons, output_field_name):
        with open(f'{opts.output}/{name}.json', 'w') as f:
            json.dump(dump, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--task', required=True, choices=['visualcomet', 'eoc'])
    parser.add_argument('--bert', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()

    main(args)
