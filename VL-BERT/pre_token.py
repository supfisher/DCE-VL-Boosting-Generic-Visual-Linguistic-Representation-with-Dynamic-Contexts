import torch
import os
from external.pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('./model/pretrained_model/bert-base-uncased')
token_path = os.path.join('./data/en_corpus/', "pretrain_tokens.pt")

import time


class GeneralCorpus(Dataset):
    def __init__(self, ann_file, encoding):
        corpus = []
        time_beg = time.time()
        for ann_file in ann_file.split('+'):
            with open(ann_file, 'r', encoding=encoding) as f:
                corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in f.readlines()])

        self.corpus = [l.strip() for l in corpus if l.strip() != '']


        print("Finished loading raw text. Costing time: ", time.time() - time_beg)

    @property
    def data_names(self):
        return ['text', 'mlm_labels']

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return tokenizer.tokenize(self.corpus[item])

    def load_corpus(self, ann_file, encoding):
        if not os.path.exists(token_path):
            corpus = []
            for ann_file in ann_file.split('+'):
                with open(ann_file, 'r', encoding=encoding) as f:
                    corpus.extend([l.strip('\n').strip('\r').strip('\n') for l in f.readlines()])

            corpus = [l.strip() for l in corpus if l.strip() != '']

            print("Finished loading raw text.")

            token_list = []
            for raw_corpu in corpus:
                token_list.append(tokenizer.tokenize(raw_corpu))

            token_tensor = torch.stack(token_list)

            torch.save(token_tensor, token_path)

            return token_tensor
        else:
            return torch.load(token_path)



encoding = 'utf-8'
ann_file = './data/en_corpus/bc1g.doc+./data/en_corpus/wiki.doc'

corpus_dataset = GeneralCorpus(ann_file, encoding)

# corpus_loader = torch.utils.data.DataLoader(dataset=corpus_dataset,
#                                          batch_size=128,
#                                          num_workers=4,
#                                          pin_memory=False,
#                                          )
#
#
# token_list = []
# for token in tqdm(corpus_loader):
#     token_list.extend(token)
#
#
# token_tensor = torch.stack(token_list)
# torch.save(token_tensor, token_path)
