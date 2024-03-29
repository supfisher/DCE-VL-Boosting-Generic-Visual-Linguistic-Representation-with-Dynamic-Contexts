"""
    Train the co-relations between two sentence.
    Generated by Guoqing Ma in KAUST.
"""
import os
import torch
import torch.nn as nn
import re
import math
from torch.optim import lr_scheduler
from torch.utils.data import SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
import json
from tqdm import tqdm
try:
    import pickle5 as pickle
except:
    import pickle

import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./mlp')


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--saved_dir", default='cache/event_similarity_mlp', type=str,
                        help="saved dir.")
parser.add_argument("--batch_size", default=32, type=int,
                        help="batch_size.")
parser.add_argument("--lr", default=1e-1, type=float,
                        help="learning rate.")
parser.add_argument("--temperature", default=1, type=float,
                        help="contrastive temperature.")
parser.add_argument("--num_epochs", default=1000, type=int,
                        help="number of epochs.")
parser.add_argument("--in_dim", default=512, type=int,
                        help="encoder in_dimision.")
parser.add_argument("--cuda", default=True, action='store_true',
                        help="if added, enable cuda.")
parser.add_argument("--logging_steps", default=2000, type=int,
                        help="logging_steps.")
parser.add_argument("--saving_steps", default=20000, type=int,
                        help="saving steps.")
parser.add_argument("--load_model_path", default=None, type=str,
                        help="load model from path.")
parser.add_argument("--start_epoch", default=0, type=int,
                        help="start epoch.")
parser.add_argument("--hvd", action='store_true',
                        help="if added, use hvd.")
parser.add_argument("--local_rank", default=-1, type=int,
                        help="local rank.")
args = parser.parse_args()
print("args: ", args)
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
args.device = device

if args.hvd:
    import horovod.torch as hvd
    hvd.init()
    args.local_rank = hvd.rank()

    torch.cuda.set_device(hvd.local_rank())
    print("hvd.local_rank():  ", hvd.local_rank(), " torch.cuda.device_count(): ", torch.cuda.device_count())
    device = torch.device(f"cuda:{hvd.local_rank()}")



# class Encoder(nn.Module):
#     def __init__(self, in_dims):
#         super(Encoder, self).__init__()
#         self.w1 = nn.Linear(in_dims, in_dims, bias=False)
#         self.w2 = nn.Linear(in_dims, in_dims, bias=False)
#         self.norm = lambda x: x.data.norm(p=2, dim=1).unsqueeze(dim=1)
#
#     def pair_loss(self, x, y, temperature=1.0, labels=None):
#         new_x = x.clone()
#         new_y = y.clone()
#         new_x[labels == -1] = y[labels == -1]
#         new_y[labels == -1] = x[labels == -1]
#         batch_size = x.shape[0]
#         x, y = self.w1(new_x), self.w2(new_y)
#         x, y = x / self.norm(x), y / self.norm(y)
#         x = x.repeat([1, batch_size]).reshape(-1, x.shape[1])
#         y = y.repeat([batch_size, 1])
#         score = (x*y).sum(dim=1).reshape(batch_size, -1)
#
#         negatives_mask = (~torch.eye(batch_size, batch_size, dtype=bool)).float().to(x.device)
#         positives = torch.diag(score, 0)
#         nominator = torch.exp(positives / temperature)
#         denominator = (negatives_mask * torch.exp(score / temperature))
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         loss = loss_partial.mean()
#         return score, loss
class Encoder(nn.Module):
    def __init__(self, in_dims):
        super(Encoder, self).__init__()
        self.bilinear = nn.Bilinear(in_dims, in_dims, 1, bias=False)
        self.norm = lambda x: x.data.norm(p=2, dim=1).unsqueeze(dim=1)

    def pair_loss(self, x, y, temperature=1.0, labels=None):
        new_x = x.clone()
        new_y = y.clone()
        new_x[labels == -1] = y[labels == -1]
        new_y[labels == -1] = x[labels == -1]
        batch_size = x.shape[0]
        x = new_x.repeat([1, batch_size]).reshape(-1, x.shape[1])
        y = new_y.repeat([batch_size, 1])
        score = self.bilinear(x, y).reshape(batch_size, -1)

        negatives_mask = (~torch.eye(batch_size, batch_size, dtype=bool)).float().to(x.device)
        positives = torch.diag(score, 0)
        nominator = torch.exp(positives / temperature)
        denominator = (negatives_mask * torch.exp(score / temperature))
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = loss_partial.mean()
        return score, loss


class EventContrastiveLoss(nn.Module):
    def __init__(self, encoder, temperature=.1, verbose=False, args=None, symmetry=1):
        super(EventContrastiveLoss, self).__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.temperature = temperature

        self.verbose = verbose

        self.encoder = encoder

        self.symmetry = symmetry

    def forward(self, emb_i, emb_j, labels):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        scores, loss = self.encoder.pair_loss(emb_i, emb_j, self.temperature, labels)
        return scores, loss


class EventDatasetPreparetion:
    def __init__(self, cache='cache/', vcg_dir='/ibex/scratch/mag0a/Github/visual-comet/data/visualcomet_annotations/', split='train'):
        super(EventDatasetPreparetion, self).__init__()
        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

        self.vcg_dir = vcg_dir
        self.split = split
        self.embed_cache = os.path.join(cache, 'event_embedding_{}.pkl'.format(self.split))
        self.sample_cache = os.path.join(cache, 'event_sample_{}.txt'.format(self.split))
        self.meta_cache = os.path.join(cache, 'meta_{}.pkl'.format(self.split))
        self.samples = []
        if os.path.exists(self.sample_cache):
            with open(self.sample_cache, 'r') as f:
                for line in f.readlines():
                    self.samples.append(line.split(','))
        else:
            records = self.load_records()
            output_pos = self.generate_pos(records)

            with open(self.sample_cache, 'w', encoding='utf-8') as f:
                for sample in output_pos:
                    f.writelines(','.join(str(s) for s in sample) + '\n')
            self.samples = output_pos
            with open(self.meta_cache, 'w') as f:
                f.writelines('{}: length of output pos {}'.format(self.split, len(output_pos)))

    def load_records(self):
        split_filename = '{}_annots.json'.format(self.split)
        vcg_path = os.path.join(self.vcg_dir, split_filename)
        print("Loading vcg dataset {}".format(vcg_path))
        # records = read_and_parse_finetune_json(vcg_path)
        with open(vcg_path, 'rb') as f:
            records = json.load(f)
        return records

    def generate_pos(self, records):
        """
            The positive samples only consider the correct orders
                (1): <before, event>
                (2): <event, after>
        """
        f = lambda str: re.sub("\d+", "person", str)
        output_pos = []
        for record in tqdm(records, "POS records: "):
            event = f(record['event'])
            if len(record['intent']) > 0:
                event += ' to ' + f(record['intent'][0])
                for intent in record['intent'][1:]:
                    event += ' and ' + f(intent)

            for after in record['after']:
                output_pos.append((event, f(after), 1))

            for before in record['before']:
                output_pos.append((event, f(before), -1))

        return output_pos

    def load_embeds(self):
        """
            load embeddings from path. If path not exists, the generate it
        """
        path = self.embed_cache
        if not os.path.exists(path):
            import tensorflow_hub as hub
            embed = hub.KerasLayer(self.module_url)

            left_sample, right_sample, labels = [], [], []
            with open(self.sample_cache, 'r') as f:
                for line in f.readlines():
                    sample = line.split(',')
                    left_sample.append(sample[0])
                    right_sample.append(sample[1])
                    labels.append(int(sample[2]))

            batch_size = 256
            len_texts = len(labels)
            embeddings = np.zeros([len_texts, 2, 512])
            for i, idx in enumerate(tqdm(range(0, len_texts, batch_size), "Embeds: ")):
                embeddings[idx:min(idx+batch_size, len_texts), 0] = embed(left_sample[idx:min(idx+batch_size, len_texts)])
                embeddings[idx:min(idx+batch_size, len_texts), 1] = embed(right_sample[idx:min(idx+batch_size, len_texts)])

            with open(path, 'wb') as handle:
                pickle.dump(
                    {
                        'embeddings' : torch.tensor(embeddings, dtype=torch.float),
                        'labels': torch.tensor(labels, dtype=torch.long)
                    },
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
            return torch.tensor(embeddings, dtype=torch.float), torch.tensor(labels, dtype=torch.long)
        else:
            with open(path, 'rb') as handle:
                samples = pickle.load(handle)
                embeddings = samples['embeddings']
                labels = samples['labels']
            return embeddings, labels


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, cache='cache', split='train'):
        super(EventDataset, self).__init__()
        prepared_dataset = EventDatasetPreparetion(cache=cache, split=split)
        self.embeddings, self.labels = prepared_dataset.load_embeds()
        self.labels[self.labels == 0] = -1

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def get_dataloader(args):
    train_dataset = EventDataset(cache=args.saved_dir, split='train')
    val_dataset = EventDataset(cache=args.saved_dir, split='val')
    print("Len of train_dataset: {}, val_dataset: {}".format(len(train_dataset), len(val_dataset)))

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, drop_last=True, sampler=train_sampler)

    val_sampler = RandomSampler(val_dataset) if args.local_rank == -1 else DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, drop_last=True, sampler=val_sampler)

    return train_loader, val_loader


def train(train_loader, model, optimizer, epoch):
    mean_loss = 0
    mean_acc = 0
    model.train()

    for batch_idx, (embeds, labels) in enumerate(train_loader):
        embeds = embeds.to(device)
        optimizer.zero_grad()

        scores, loss = model(embeds[:, 0], embeds[:, 1], labels)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        mean_acc += metrics(scores, torch.arange(embeds.shape[0]).to(device))

        if batch_idx % args.logging_steps == 0 and args.local_rank in [-1, 0]:
            print('TRAIN:: Epoch: {}  [{}/{} ({:.0f}%)]\t Loss: {:.6f} \t acc: {:.3f}\t'.format(
                    epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                            100. * batch_idx / len(train_loader), mean_loss/(batch_idx+1), mean_acc/(batch_idx+1)))

    print('TRAIN:: Epoch: {}  [{}/{} ({:.0f}%)]\t Loss: {:.6f} \t acc: {:.3f}\t'.format(
        epoch, batch_idx * args.batch_size, len(train_loader.dataset),
               100. * batch_idx / len(train_loader), mean_loss / (batch_idx + 1), mean_acc/(batch_idx+1)))
    return mean_loss / (batch_idx + 1), mean_acc/(batch_idx+1)


def evaluate(test_loader, model, epoch):
    mean_loss = 0
    mean_acc = 0
    model.eval()
    for batch_idx, (embeds, labels) in enumerate(test_loader):
        embeds = embeds.to(device)

        scores, loss = model(embeds[:, 0], embeds[:, 1], labels)
        mean_loss += loss.item()
        mean_acc += metrics(scores, torch.arange(embeds.shape[0]).to(device))

    print('EVAL:: Epoch: {}   [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t acc: {:.3f}\t'.format(
                    epoch, batch_idx * args.batch_size, len(test_loader.dataset),
                                                        100. * batch_idx / len(test_loader), mean_loss/(batch_idx+1), mean_acc/(batch_idx+1)))
    return mean_loss/(batch_idx+1), mean_acc/(batch_idx+1)


def metrics(pred_logits, labels, k=4):
    pred_labels = pred_logits.topk(k, dim=1, largest=True)
    count = 0
    for preds, label in zip(pred_labels, labels):
        if label in preds:
            count += 1
    acc = count/len(pred_labels)
    return acc



def writer_record(train_loss, train_acc, val_loss, val_acc, epoch):
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Acc/train", train_acc, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)


if __name__ == "__main__":
    train_loader, val_loader = get_dataloader(args)
    encoder = Encoder(in_dims=args.in_dim)
    corr_net = EventContrastiveLoss(encoder, temperature=args.temperature, args=args, symmetry=-1).to(device)

    optimizer = torch.optim.Adam(corr_net.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.load_model_path is not None:
        model_CKPT = torch.load(args.load_model_path)
        corr_net.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        # optimizer.load_state_dict(model_CKPT['optimizer'])
        args.start_epoch = model_CKPT['epoch']

    if args.local_rank != -1:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=corr_net.named_parameters())

        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(corr_net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    current_lr = args.lr
    T_max = 100
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=current_lr * 0.1)
    best_acc = 0
    for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
        if epoch % T_max == 0:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=current_lr * 0.1)
            current_lr = current_lr * 0.1
        scheduler.step()

        train_loss, train_acc= train(train_loader, corr_net, optimizer, epoch)
        val_loss, val_acc = evaluate(val_loader, corr_net, epoch)

        writer_record(train_loss, train_acc, val_loss, val_acc, epoch)

        if best_acc < val_acc:
            torch.save({'epoch': epoch + 1, 'state_dict': corr_net.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.saved_dir, "corr_net_best.pth"))
            best_acc = val_acc

        print("------------------------------------------------------------")
        if args.local_rank in [-1, 0]:
            print("Optimizer learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])

            if epoch % 10 == 0 and args.saved_dir is not None:

                torch.save({'epoch': epoch + 1, 'state_dict': corr_net.state_dict(), 'optimizer': optimizer.state_dict()},
                               os.path.join(args.saved_dir, "corr_net_{}.pth".format(epoch)))
