import codecs
import os
import random
import xml.etree.ElementTree as ET
import numpy as np

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

from common.utils.create_logger import makedirsExist
from common.utils.zipreader import ZipReader
from external.pytorch_pretrained_bert import BertTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


ImagePath='/ibex/scratch/mag0a/Github/VL-BERT/data/flickr30/flickr30k_images/flickr30k_images'
DataPath = '/ibex/scratch/mag0a/Github/VL-BERT/data/flickr30/flickr30k_entities/'


class FlickrITDataset(Dataset):
    def __init__(self, root_path, data_path=DataPath, image_set=ImagePath, split='test',
                 seq_len=64, transform=None, test_mode=False, with_rel_task=True,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, **kwargs):
        """
        Conceptual Captions Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr_bakup dataset
        :param transform: transform
        :param test_mode: EBS_VT mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(FlickrITDataset, self).__init__()

        annot = {'train': 'train.txt',
                 'val': 'val.txt',
                 'test': 'test.txt'}

        self.seq_len = seq_len
        self.split = split
        self.data_path = data_path
        self.root_path = root_path
        self.with_rel_task = with_rel_task
        self.ann_file = os.path.join(data_path, annot[split])

        self.image_set = image_set
        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')
        self.add_image_as_a_box = add_image_as_a_box
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        self.zipreader = ZipReader()

        with codecs.open(self.ann_file, encoding="utf-8") as f:
            self.database = [l.strip('\n') for l in f.readlines()]

        assert self.split == 'test', "We only work on zero-shot ITR"
        self.num_captions_per_img = len(self.database)

    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'text', 'img_index', 'txt_index']

    def get_text(self, index):
        idb = self.database[index // self.num_captions_per_img // 5]
        sentences = get_sentence_data(os.path.join(self.data_path, 'Sentences/{}.txt'.format(idb)))
        caption = sentences[index % 5]

        caption_tokens = self.tokenizer.tokenize(' '.join(caption))
        text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        text = self.tokenizer.convert_tokens_to_ids(text_tokens)
        return text

    def get_img(self, index):
        idb = self.database[index // self.num_captions_per_img // 5]
        annotation = get_annotations(os.path.join(self.data_path, 'Annotations/{}.xml'.format(idb)))
        # image data
        boxes = torch.as_tensor(annotation['boxes'])

        image = self._load_image(os.path.join(self.image_set, '{}.jpg'.format(idb)))
        w0, h0 = image.size

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1.0, h0 - 1.0]])
            boxes = torch.cat((image_box, boxes), dim=0)

        # transform
        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        return image, boxes, im_info

    def __getitem__(self, index):
        img_index = index // (self.num_captions_per_img * 5)
        txt_index = index % (self.num_captions_per_img * 5)
        image, boxes, im_info = self.get_img(index)
        text = self.get_text(index)
        # truncate seq to max len
        if len(text) + len(boxes) > self.seq_len:
            text_len_keep = len(text)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len and (text_len_keep > 0) and (box_len_keep > 0):
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            if box_len_keep < 1:
                box_len_keep = 1
            boxes = boxes[:box_len_keep]
            text = text[:(text_len_keep - 1)] + [text[-1]]

        return image, boxes, im_info, text, img_index, txt_index

    def __len__(self):
        return len(self.database)*self.num_captions_per_img*5

    def _load_image(self, path):
        return Image.open(path).convert('RGB')


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset
    input:
      fn - full file path to the annotations file to parse
    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': [], 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset
    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to
    """
    with codecs.open(fn, encoding="utf-8") as f:
        sentences = [l.strip('\n') for l in f.readlines()]

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        # sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        # for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
        #     sentence_data['phrases'].append({'first_word_index': index,
        #                                      'phrase': phrase,
        #                                      'phrase_id': p_id,
        #                                      'phrase_type': p_type})

        annotations.append(' '.join(words))

    return annotations


