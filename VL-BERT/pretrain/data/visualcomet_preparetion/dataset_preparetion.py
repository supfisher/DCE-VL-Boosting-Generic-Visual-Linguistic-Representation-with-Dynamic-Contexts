"""
    Extract the event records from raw dataset. Reformat it according to <before, now, 1> and <now, after, 1>,
    and store the formated setences into 'cache/event_sample_{}.txt'.format(split). Then apply google USE to
    embeds the cached setence into tensors, and save them into 'cache/event_embedding_{}.pkl'.format(split)

    Generated by Guoqing Ma in KAUST.
"""

import os
import math
# from utils.file_utils import read_and_parse_finetune_json
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import re
try:
    import pickle5 as pickle
except:
    import pickle

import argparse
parser = argparse.ArgumentParser()


class RecordParseDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', cache='cache/event_similarity', load_cache=True):
        '''
            parse the records, and generate the (image, event) pairs
        '''
        super(RecordParseDataset, self).__init__()
        self.split = split
        self.load_cache = load_cache
        self.parse_record_cache = os.path.join(cache, '{}_record_img_event.txt'.format(split))
        self.embed_cache = os.path.join(cache, '{}_use_embedding.pkl'.format(self.split))
        self.dataset = []
        if not os.path.exists(self.parse_record_cache) or not load_cache:
            records = self.load_records(self.split)
            f = lambda str: re.sub("\d+", "[person]", str)
            for record in tqdm(records, "Records: "):
                event = f(record['event'])
                if len(record['intent']) > 0:
                    for intent in record['intent']:
                        event += '; [intent] ' + f(intent)
                if 'place' in record:
                    event += '; [place] ' + f(record['place'])

                self.dataset.append((record['img_fn'], record['metadata_fn'], event))

            with open(self.parse_record_cache, 'w', encoding="utf-8") as f:
                for sample in self.dataset:
                    f.writelines(','.join(str(s) for s in sample) + '\n')
        else:
            with open(self.parse_record_cache, 'r', encoding="utf-8") as f:
                for l in f.readlines():
                    self.dataset.append(l.split(","))

        self.event_embeddings = self.load_embeds()

    def load_records(self, split):
        vcg_dir = '/ibex/scratch/mag0a/Github/visual-comet/data/visualcomet_annotations/'
        split_filename = '{}_annots.json'.format(split)
        vcg_path = os.path.join(vcg_dir, split_filename)
        print("Loading vcg dataset {}".format(vcg_path))
        # records = read_and_parse_finetune_json(vcg_path)
        with open(vcg_path, 'rb') as f:
            records = json.load(f)
        return records

    # def load_records(self, split):
    #     '''
    #     :param split: whether 'train' or 'val'
    #     :return: loading raw records from database, return the raw records list
    #     '''
    #     self.vcg_dir = '/ibex/scratch/mag0a/Github/visual-comet/data/visualcomet_annotations/'
    #     split_filename = '{}_annots.json'.format(split)
    #     vcg_path = os.path.join(self.vcg_dir, split_filename)
    #     print("Loading vcg dataset {}".format(vcg_path))
    #     records = read_and_parse_finetune_json(vcg_path)
    #
    #     return records

    def load_embeds(self):
        """
            load pre-trained event embeddings from path. If path not exists, then pre train it
        """
        path = self.embed_cache
        if not os.path.exists(path) or not self.load_cache:
            import tensorflow_hub as hub
            embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-large/5")

            imgs, events = [], []
            with open(self.parse_record_cache, 'r', encoding="utf-8") as f:
                for line in f.readlines():
                    img_event = line.split(',')
                    imgs.append(img_event[0])
                    events.append(img_event[-1])

            batch_size = 256
            len_events = len(events)
            event_embeddings = np.zeros([len_events, 512])
            for i, idx in enumerate(tqdm(range(0, len_events, batch_size), "Embeds: ")):
                event_embeddings[idx:min(idx + batch_size, len_events)] = embed(
                    events[idx:min(idx + batch_size, len_events)])

            torch.save(
                {
                    'event_embeddings': torch.tensor(event_embeddings, dtype=torch.float),
                },
                path)

            return torch.tensor(event_embeddings, dtype=torch.float)
        else:
            event_embeddings = torch.load(path)['event_embeddings']
            return event_embeddings


class ImgPostivePairs(torch.utils.data.Dataset):
    """
        generate positive image pairs according to event based similarity
    """
    def __init__(self, split='train',
                 cache="cache/event_similarity/",
                 load_cache=True,
                 simi_trick='dcbs'):
        super(ImgPostivePairs, self).__init__()
        self.simi_trick = simi_trick
        self.split = split
        record_parse_dataset = RecordParseDataset(split=split, cache=cache, load_cache=load_cache)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.event_embeddings = record_parse_dataset.event_embeddings.to(self.device)

        ## preprocess the event_embeddings matmul corr_mat, and save it.
        if simi_trick=='dcbs':
            coeffi_CKPT = torch.load(os.path.join(cache, 'accumulated_scores_best.pth'))
            self.coeffi_left = self.dis_abnorm(coeffi_CKPT["{}_accumulated_scores_left".format(self.split)]).mean()**2
            self.coeffi_right = self.dis_abnorm(coeffi_CKPT["{}_accumulated_scores_right".format(self.split)]).mean()**2

            encoder_cache = os.path.join(cache, '{}_event_embeds_matmul_corrmat.pkl'.format(self.split))
            if not os.path.exists(encoder_cache) or not load_cache:
                model_CKPT = torch.load(os.path.join(cache, 'corr_mat_best.pth'))
                corr_mat = model_CKPT['corr_mat'].squeeze().to(self.device)

                self.encoder_left = self.event_embeddings.matmul(corr_mat.T)
                self.encoder_right = self.event_embeddings.matmul(corr_mat)
                torch.save(
                    {
                        'encoder_left': self.encoder_left,
                        'encoder_right': self.encoder_right,
                    },
                    encoder_cache)

            else:
                encoder = torch.load(encoder_cache)
                self.encoder_left = encoder['encoder_left']
                self.encoder_right = encoder['encoder_right']

        ## pregenerate the positive pairs, and save it..
        positive_pairs_cache = os.path.join(cache, '{}_{}_positive_pairs.pkl'.format(simi_trick, self.split))
        if not os.path.exists(positive_pairs_cache) or not load_cache:
            self.positive_pairs = self.generate_positive_pairs()
            torch.save(
                {
                    'positive_pairs': self.positive_pairs.detach().cpu(),
                },
                positive_pairs_cache
            )
        else:
            self.positive_pairs = torch.load(positive_pairs_cache)['positive_pairs']

    def dis_abnorm(self, coeffi):
        coeffi[coeffi < 0] = 0
        coeffi[coeffi > 30] = 0
        return coeffi

    def similarity_mat(self, mat1, mat2, coefficient=1):
        similarity = F.cosine_similarity(mat1.unsqueeze(1), mat2.unsqueeze(0), dim=2)
        similarity *= coefficient
        return similarity

    def event_based_similarity(self, batch_idx):
        if self.simi_trick == 'dcbs':
            similarity_matrix_mid = self.similarity_mat(self.event_embeddings[batch_idx], self.event_embeddings)

            similarity_matrix_left = self.similarity_mat(self.encoder_left[batch_idx], self.encoder_left, self.coeffi_left)

            similarity_matrix_right = self.similarity_mat(self.encoder_right[batch_idx], self.encoder_right, self.coeffi_right)

            similarity_matrix = (similarity_matrix_mid + similarity_matrix_left + similarity_matrix_right) / 3.0
        elif self.simi_trick == 'cosine':
            similarity_matrix = self.similarity_mat(self.event_embeddings[batch_idx], self.event_embeddings)
        else:
            raise NotImplementedError

        similarity_matrix[range(len(batch_idx)), batch_idx] = 0.0
        return similarity_matrix

    def get_positive_index(self, similarity_mat):
        return similarity_mat.argmax(dim=1)

    def generate_positive_pairs(self):
        positive_pairs = []
        bs = 32
        for i in tqdm(range(0, len(self.event_embeddings), bs), "Positive"):
            batch_idx = torch.arange(i, min(i+bs, len(self.event_embeddings))).to(self.device)
            similarity_matrix = self.event_based_similarity(batch_idx)
            positive_index = self.get_positive_index(similarity_matrix)
            positive_pairs.append(positive_index)
        return torch.cat(positive_pairs, dim=0)

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        return index, self.positive_pairs[index]



if __name__ == '__main__':
    parser.add_argument("--saved_dir", default='cache/event_similarity_cleaned', type=str,
                        help="saved dir.")
    parser.add_argument("--simi_trick", default='dcbs', type=str,
                        help="similarity trick.")

    args = parser.parse_args()
    print("args: ", args)


    print("Finished EBS dataset preparation, generated event positive pairs and negative pairs datasets for train and val... ")

    for split in ['train', 'val']:
        img_positive_pairs_generator = ImgPostivePairs(split=split, cache=args.saved_dir, load_cache=True, simi_trick=args.simi_trick)

        print("Generated image positive pairs according to EBS given the pre-generated event positive pairs... ")

        print()