"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import os, json, copy
import numpy as np
from .data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)
from .data import get_ids_and_lens
from .itm import ItmDataset
from .vcr import VcrDetectFeatTxtTokDataset


record_cache = '/ibex/scratch/mag0a/Github/VL-BERT/pretrain/data/visualcomet_preparetion/cache/event_similarity_cleaned/'


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask


def _get_img_tgt_mask(img_mask, txt_len):
    z = torch.zeros(txt_len, dtype=torch.uint8)
    img_mask_tgt = torch.cat([z, img_mask], dim=0)
    return img_mask_tgt


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


class EocDataset(DetectFeatTxtTokDataset):
    """ visual comet event order classification dataset """
    def __init__(self, txt_db, img_db, neg_sample_p=0.5):
        super().__init__(txt_db, img_db)
        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids)*2,
            p=[self.neg_sample_p, 1-self.neg_sample_p])

    def __len__(self):
        return len(self.ids)*2

    def __getitem__(self, i):
        id_ = self.ids[i//2]
        example = self.txt_db[id_]

        input_ids = example['input_ids']
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        if ground_truth_label == 1:
            ground_truth_label = example['label']

        # text input
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return torch.tensor(input_ids), img_feat, img_pos_feat, attn_masks, target


class ConDataset(DetectFeatTxtTokDataset):
    """
        Contrastive learning
    """
    def __init__(self, txt_db, img_db, is_train=True):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        split = 'train' if is_train else 'val'
        positive_pairs_cache = os.path.join(record_cache, 'dcbs_{}_positive_pairs.pkl'.format(split))
        assert os.path.exists(
            positive_pairs_cache), "positive_pairs_cache needs to be existed, plese generate it first: call ImgPostivePairs"

        positive_pairs = torch.load(positive_pairs_cache)
        self.positive_pairs = positive_pairs['positive_pairs']
        self.mask_prob = 0.15

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        p_id_ = str(self.positive_pairs[int(id_)].item())
        p_example = self.txt_db[p_id_]
        return self.load_data(example) + self.load_data(p_example)

    def load_data(self, example):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """

        input_ids = example['input_ids']

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return torch.tensor(input_ids), img_feat, img_pos_feat, attn_masks, img_mask, img_mask_tgt


def con_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    len_batch = len(inputs)
    len_data = len(inputs[0]) // 2
    for i in range(len_batch):
        inputs.append(inputs[i][len_data:])
        inputs[i] = inputs[i][:len_data]

    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt}
    return batch


class OodDataset(DetectFeatTxtTokDataset):
    """
    Out of domain (image-text) similarity analysis
    """
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids = example['input_ids']

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return torch.tensor(input_ids), img_feat, img_pos_feat, attn_masks, example['img_fname']


def ood_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, img_fname) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_fname': img_fname}
    return batch


class OodVcrDataset(VcrDetectFeatTxtTokDataset):
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split = split
        assert self.task == "qa,qar",\
            "loading evaluation dataset with two tasks together"

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]

        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'][0], example['img_fname'][1])

        input_ids = torch.tensor(example['input_ids'])
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        # attn_masks = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, example['img_fname'][0]


# def ood_collate(inputs):
#     """
#     Return:
#     :input_ids    (n, max_L) padded with 0
#     :position_ids (n, max_L) padded with 0
#     :txt_lens     list of [txt_len]
#     :img_feat     (n, max_num_bb, feat_dim)
#     :img_pos_feat (n, max_num_bb, 7)
#     :num_bbs      list of [num_bb]
#     :attn_masks   (n, max_{L + num_bb}) padded with 0
#     """
#     (input_ids, img_feats, img_pos_feats, attn_masks, img_fname) = map(list, unzip(inputs))
#
#     # text batches
#     txt_lens = [0 for i in input_ids]
#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                 ).unsqueeze(0)
#
#     # image batches
#     num_bbs = [f.size(0) for f in img_feats]
#     img_feat = pad_tensors(img_feats, num_bbs)
#     img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
#
#     attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
#
#     bs, max_tl = input_ids.size()
#     out_size = attn_masks.size(1)
#     gather_index = get_gather_index(txt_lens, num_bbs, bs, 0, out_size)
#
#     batch = {'input_ids': input_ids,
#              'position_ids': position_ids,
#              'img_feat': img_feat,
#              'img_pos_feat': img_pos_feat,
#              'attn_masks': attn_masks,
#              'gather_index': gather_index,
#              'img_fname': img_fname}
#     return batch



def convert_str_mins(time):
    time_list = time.split('.')
    assert len(time_list) == 4
    time = int(time_list[-1]) + int(time_list[-2]) * 60 + int(time_list[-3]) * 3600
    return time


def parse_file_name(file_name):
    ori_file_name = file_name
    file_name = file_name.split('@')
    at_index = file_name[-1].split('.')[0]
    file_name = file_name[0].split('-')
    time_end = file_name[-1]
    file_name = file_name[-2]
    time_begin = file_name[-len("00.17.26.000"):]
    try:
        return convert_str_mins(time_begin), convert_str_mins(time_end), at_index
    except:
        print(ori_file_name)


def is_in_one_cluster(file_prev, file):
    prev_time_begin, prev_time_end, prev_at_index = parse_file_name(file_prev)
    time_begin, time_end, at_index = parse_file_name(file)
    if time_begin == prev_time_begin:
        return True
    else:
        return False


def sample_one_cluster(one_cluster):
    samples = []
    one_cluster = sorted(one_cluster, key=lambda x: int(x.split('@')[1].split('.')[0]))
    if len(one_cluster) > 1:
        for i, img in enumerate(one_cluster):
            for neg_img in one_cluster[:i]:
                samples.append([img, neg_img, 0])
            for pos_img in one_cluster[i + 1:]:
                samples.append([img, pos_img, 1])
    return samples


def gen_image_clusters(path):
    image_clusters = []
    g = os.walk(path)
    for path, dir_list, file_list in g:
        for dir_name in dir_list:
            files = [os.path.join(dir_name, f) for f in os.listdir(os.path.join(path, dir_name)) if
                     ('jpg' in f) and ('npy' not in f)]
            files = sorted(files)
            file_prev = files[0]
            one_cluster = [file_prev]
            if 'lsmdc' in dir_name:  # only lsmdc dir has time stamp
                for file in files[1:]:
                    if is_in_one_cluster(file_prev, file):
                        one_cluster.append(file)
                    else:
                        image_clusters.append(one_cluster)
                        one_cluster = [file]
                    file_prev = file
            else:
                for file in files[1:]:
                    if file_prev.split('@')[0] == file.split('@')[0]:
                        one_cluster.append(file)
                    else:
                        image_clusters.append(one_cluster)
                        one_cluster = [file]
                    file_prev = file

    image_clusters = [one_cluster for one_cluster in image_clusters if len(one_cluster) >= 2]
    return image_clusters


def gen_image_ordering_cache(image_clusters):
    image_ordering_cache = []
    for one_cluster in image_clusters:
        samples = sample_one_cluster(one_cluster)
        image_ordering_cache.extend(samples)
    return image_ordering_cache


class IocDataset(DetectFeatTxtTokDataset):
    """
        Image ordering classification
    """
    def __init__(self, txt_db, img_db, is_train=True, mask_prob=0):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(self.txt_db)
        self.img_ids = self.img_fn_mapping_ids()

        path = "/ibex/scratch/mag0a/Github/visual-comet/data/vcr1images"
        if os.path.exists('image_cluster.json'):
            with open('image_cluster.json', 'r') as f:
                image_clusters = json.load(f)
        else:
            image_clusters = gen_image_clusters(path)
            with open('image_cluster.json', 'w') as f:
                json.dump(image_clusters, f)

        self.image_ordering_one_cluster = []
        self.image_ordering_cache = []
        left_images_len = 0

        for one_cluster in image_clusters:
            samples = sample_one_cluster(one_cluster)
            new_samples = []
            for img_i, img_j, ordering_label in samples:
                id_i, id_j = img_i.split('/')[-1], img_j.split('/')[-1]
                if id_i in self.img_ids.keys() and id_j in self.img_ids.keys():
                    new_samples.append([img_i, img_j, ordering_label])
            if len(new_samples) > 2:
                self.image_ordering_one_cluster.append(new_samples)
                self.image_ordering_cache.extend(sorted(new_samples)[:len(new_samples)//2])
                left_images_len += len(new_samples)//2

        self.lens = [32 for _ in range(len(self.image_ordering_cache))]
        self.len_clusters = len(self.image_ordering_one_cluster)
        self.mask_prob = 0.0
        print("len of image_ordering_cache: {}, left_images_len: {}".format(len(self.image_ordering_cache), left_images_len))

    def __len__(self):
        return len(self.image_ordering_cache)

    def img_fn_mapping_ids(self):
        img_ids = {}
        for id_ in self.ids:
            example = self.txt_db[id_]
            img_fn = example['img_fname']
            if img_fn in img_ids:
                img_ids[img_fn].append(id_)
            else:
                img_ids[img_fn] = [id_]
        return img_ids

    def get_one_cluster(self, samples):
        def yied_one(samples):
            for id_i, id_j, ordering_label in samples:
                id_i, id_j = id_i.split('/')[-1], id_j.split('/')[-1]
                yield self.load_img(id_i) + self.load_img(id_j) + (ordering_label,)
        ordering = sorted(set(list(zip(*samples))[0]), key=lambda x: int(x.split('@')[1].split('.')[0]))

        return list(yied_one(samples)), ordering

    def __getitem__(self, i):
        id_i, id_j, ordering_label = self.image_ordering_cache[i]
        id_i, id_j = id_i.split('/')[-1], id_j.split('/')[-1]
        return self.load_img(id_i) + self.load_img(id_j) + (ordering_label, )

    def __gen_image_order(self, img_fn):
        """Generate the order of an imgae"""
        return int(img_fn.split('@')[1].split('.')[0])

    def __gen_nearest_input_ids(self, img_fn):
        distance = 10000
        nearst_input_ids = None
        pre_img_fn = None
        for id_ in self.img_ids[img_fn.split('@')[0]]:
            example = self.txt_db[id_]
            if abs(self.__gen_image_order(img_fn)-self.__gen_image_order(example['img_fname'])) <= distance:
                distance = abs(self.__gen_image_order(img_fn)-self.__gen_image_order(example['img_fname']))
                nearst_input_ids = torch.tensor(example['input_ids'])
                # if pre_img_fn == example['img_fname']:
                #     nearst_input_ids = torch.cat([nearst_input_ids, torch.tensor(example['input_ids'])], dim=0)
                # pre_img_fn = example['img_fname']
        return nearst_input_ids

    def load_img(self, img_fname):
        p_ids = self.img_ids[img_fname]
        # input_ids = self.__gen_nearest_input_ids(img_fname)
        # input_ids = torch.cat([torch.tensor(self.txt_db[pid]['input_ids']) for pid in p_ids], dim=0)
        input_ids = torch.tensor(self.txt_db[p_ids[0]]['input_ids'])
        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, 0)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        # attn_masks = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, img_mask_tgt


def ioc_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    ordering_label = torch.tensor([i[-1] for i in inputs])
    inputs = [i[:-1] for i in inputs]
    len_batch = len(inputs)
    len_data = len(inputs[0]) // 2
    for i in range(len_batch):
        inputs.append(inputs[i][len_data:])
        inputs[i] = inputs[i][:len_data]

    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    # txt_lens = [0 for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    # img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    # bs = len(img_feats)
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'ordering': ordering_label}
    return batch


class IocDatasetWOCaption(DetectFeatTxtTokDataset):
    """
        Image ordering classification
    """
    def __init__(self, txt_db, img_db, is_train=True, mask_prob=0):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        txt_lens, self.ids = get_ids_and_lens(self.txt_db)
        self.img_ids = self.img_fn_mapping_ids()

        path = "/ibex/scratch/mag0a/Github/visual-comet/data/vcr1images"
        if os.path.exists('image_cluster.json'):
            with open('image_cluster.json', 'r') as f:
                image_clusters = json.load(f)
        else:
            image_clusters = gen_image_clusters(path)
            with open('image_cluster.json', 'w') as f:
                json.dump(image_clusters, f)

        self.image_ordering_one_cluster = []
        self.image_ordering_cache = []

        for one_cluster in image_clusters:
            samples = sample_one_cluster(one_cluster)
            new_samples = []
            for img_i, img_j, ordering_label in samples:
                id_i, id_j = img_i.split('/')[-1], img_j.split('/')[-1]
                if id_i.split('@')[0] in self.img_ids.keys() and id_j.split('@')[0] in self.img_ids.keys():
                    new_samples.append([img_i, img_j, ordering_label])
            self.image_ordering_cache.extend(new_samples[:2])
            if len(new_samples) > 2:
                self.image_ordering_one_cluster.append(new_samples)

        self.lens = [32 for _ in range(len(self.image_ordering_cache))]
        self.len_clusters = len(self.image_ordering_one_cluster)
        self.mask_prob = 0.15
        print()

    def __len__(self):
        return len(self.image_ordering_cache)

    def img_fn_mapping_ids(self):
        img_ids = {}
        for id_ in self.ids:
            example = self.txt_db[id_]
            img_fn = example['img_fname']
            img_fn = img_fn.split('@')[0]
            if img_fn in img_ids:
                img_ids[img_fn].append(id_)
            else:
                img_ids[img_fn] = [id_]
        return img_ids

    def get_one_cluster(self, samples):
        def yied_one(samples):
            for id_i, id_j, ordering_label in samples:
                id_i, id_j = id_i.split('/')[-1], id_j.split('/')[-1]
                yield self.load_img(id_i) + self.load_img(id_j) + (ordering_label,)
        ordering = sorted(set(list(zip(*samples))[0]), key=lambda x: int(x.split('@')[1].split('.')[0]))

        return list(yied_one(samples)), ordering

    def __getitem__(self, i):
        id_i, id_j, ordering_label = self.image_ordering_cache[i]
        id_i, id_j = id_i.split('/')[-1], id_j.split('/')[-1]
        return self.load_img(id_i) + self.load_img(id_j) + (ordering_label, )

    def load_img(self, img_fname):
        input_ids = torch.zeros(30)
        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask_tgt = _get_img_tgt_mask(img_mask, 0)
        attn_masks = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, img_mask_tgt


def ioc_collate_wo_cap(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    ordering_label = torch.tensor([i[-1] for i in inputs])
    inputs = [i[:-1] for i in inputs]
    len_batch = len(inputs)
    len_data = len(inputs[0]) // 2
    for i in range(len_batch):
        inputs.append(inputs[i][len_data:])
        inputs[i] = inputs[i][:len_data]

    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts) = map(list, unzip(inputs))

    # text batches
    # txt_lens = [i.size(0) for i in input_ids]
    txt_lens = [0 for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    # img_feat = _mask_img_feat(img_feat, img_masks)
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    # bs = len(img_feats)
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, 0, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'ordering': ordering_label}
    return batch
