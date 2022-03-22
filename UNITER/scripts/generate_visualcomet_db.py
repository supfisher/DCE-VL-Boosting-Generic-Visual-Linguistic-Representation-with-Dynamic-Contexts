import argparse
import os
from common.module import Module
from common.fast_rcnn import FastRCNN
from PIL import Image
from torch.utils.data import Dataset
from flickr.data.transforms.build import build_transforms
from tqdm import tqdm
from common.trainer import to_cuda
import torch

from uniter_pretrain.function.config import config, update_config
from uniter_pretrain.modules.uniter import UniterForPretraining
from uniter_pretrain.data.build import make_dataloader
from uniter_pretrain.modules.model_ import UniterConfig

import lmdb

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass


class Model(UniterForPretraining):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.img_dim,
                                                enable_cnn_reg_loss=False)

    def forward(self, *inputs):
        inputs1, inputs2 = inputs[0:len(inputs)//2], inputs[len(inputs)//2:]
        output = self.pre_compute_image(*inputs1)
        return output


def train(dataloader, model, env):
    i = 0
    model.eval()
    cache = {}
    for nbatch, batch in tqdm(enumerate(dataloader), 'FlickEmbed: '):
        batch = to_cuda(batch)
        embeds = model(*batch)
        for emb in embeds:
            cache["image_{}".format(i)] = emb
            i+=1
            if i> 10:
                break

    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, bytes):
                # 图片类型为bytes
                txn.put(k.encode(), v)
            else:
                # 标签类型为str, 转为bytes
                txn.put(k.encode(), v.encode())  # 编码

    env.close()


def main(args, config):
    model = Model(UniterConfig.from_json_file(config)).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    config.TRAIN.SHUFFLE = False
    config.VAL.SHUFFLE = False
    config.TEST.SHUFFLE = False
    config.TRAIN.FP16 = False

    for split in ['train', 'val', 'test']:
        dataloader = make_dataloader(config, mode=split)

        path = os.path.join(args.saved_path, split)

        if not os.path.exists(path):
            os.makedirs(path)

        env = lmdb.open(path, map_size=1073741824)
        train(dataloader, model, env)


def parse_args():
    parser = argparse.ArgumentParser('VisualComet Embedding Network')
    parser.add_argument('--cfg', default='cfgs/uniter/pretrain/pretrain_uniter_contrastive.yaml',
                        type=str, help='path to config file')
    parser.add_argument('--saved_path', default='data/pre_embed/visualcomet', type=str, help='saved_path')

    args = parser.parse_args()
    update_config(args.cfg)
    return args, config


if __name__ == '__main__':
    args, config = parse_args()

    main(args, config)

