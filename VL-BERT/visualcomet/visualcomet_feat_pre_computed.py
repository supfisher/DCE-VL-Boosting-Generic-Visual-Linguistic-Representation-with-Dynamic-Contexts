import _init_paths
import os
import argparse
import torch
import subprocess

from visualcomet.function.config import config, update_config
from visualcomet.modules import FastRCNNForVisualComet
from visualcomet.data.build import make_dataloader
import numpy as np
from apex import amp
from apex.parallel import DistributedDataParallel as Apex_DDP
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import json
import h5py, pickle

path='pretrain/data/visualcomet_preparetion/cache/pre_computed_img_embed'


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda()
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda()

    return batch


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--log-dir', type=str, help='tensorboard log dir')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--do-EBS_VT', help='whether to generate csv result on EBS_VT set',
                        default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')
    parser.add_argument('--mode', type=str, default='test')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29521)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config


def main(args, config):
    model = eval(config.MODULE)(config).cuda()
    world_size, rank = 1, 0

    if args.dist:
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23457)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)
        if args.slurm:
            distributed.init_process_group(backend='nccl')
        else:
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)
        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if config.TRAIN.FP16:
        [model] = amp.initialize([model],
                                        opt_level='O2',
                                        keep_batchnorm_fp32=False)
        if args.dist:
            model = Apex_DDP(model, delay_allreduce=True)

    return model, world_size, rank


def pre_compute_img_feat(model, world_size, rank, split):
    model.eval()
    data_loader, data_sampler = make_dataloader(config, mode=split,
                                  distributed=args.dist,
                                  num_replicas=world_size,
                                  rank=rank,
                                  expose_sampler=True)
    len_dataset = len(data_loader.dataset)
    saved_embedding = np.zeros([len_dataset, 100, config.NETWORK.IMAGE_FINAL_DIM], dtype=np.float32)
    saved_num_bbs = np.zeros(len_dataset)
    max_num_bbs = 0
    print("rank: {} ,length of {} dataset: {}, batch size: {}".format(rank, split, len_dataset, data_loader.batch_size))

    model.eval()
    for batch in tqdm(data_loader, 'rank:{} {}_loader'.format(rank, split)):
        num_bbs = batch[-1]
        index = batch[-2]
        batch = batch[:-2]
        batch = to_cuda(batch)

        try:
            image_feat = model(*batch)
            image_feat = image_feat.detach().cpu()
            saved_embedding[index, :image_feat.shape[1]] = image_feat
            saved_num_bbs[index] = num_bbs
            max_num_bbs = max(max_num_bbs, max(num_bbs))
        except BaseException:
            print("rank: {} index: {}, num_bb: {}".format(rank, index, num_bbs))

    try:
        saved_path = "{}/pre_computed_img_feat_rank_{}.h5".format(split, rank) if args.dist else "{}/pre_computed_img_feat.h5".format(split)
        with h5py.File(os.path.join(path, saved_path), 'w') as f:
            f.create_dataset('img_feats', data=saved_embedding)
            f.create_dataset('num_bbs', data=saved_num_bbs)

        save_img_fns_path = "{}/img_fns_{}.jsonl".format(split, rank) if args.dist else "{}/img_fns.jsonl".format(split)
        with open(os.path.join(path, save_img_fns_path), 'w') as f:
            json.dump({
                "img_fns": data_loader.dataset.img_fns,
                "metadata_fns": data_loader.dataset.metadata_fns
            }, f)

    except:
        print("{} file saved error".format(split))


if __name__ == '__main__':
    args, config = parse_args()
    model, world_size, rank = main(args, config)
    pre_compute_img_feat(model, world_size, rank, args.mode)

    # "combine splitted  image feats"
    # if rank == 0 and args.dist:
    #     split = args.mode
    #     data_loader, data_sampler = make_dataloader(config, mode=split,
    #                                                 distributed=args.dist,
    #                                                 num_replicas=world_size,
    #                                                 rank=rank,
    #                                                 expose_sampler=True)
    #     len_dataset = len(data_loader.dataset)
    #     saved_embedding = np.zeros([len_dataset, 100, config.NETWORK.IMAGE_FINAL_DIM], dtype=np.float32)
    #     saved_num_bbs = np.zeros(len_dataset)
    #
    #     for i in range(4):
    #         with open(os.path.join(path, "pre_computed_img_feat_{}_rank_{}.pkl".format(split, rank)), 'rb') as f:
    #             data_dict = pickle.load(f)
    #         saved_embedding += data_dict["img_feats"]
    #         saved_num_bbs += data_dict["num_bbs"]
    #         img_fns = data_dict["img_fns"]
    #         metadata_fns = data_dict["metadata_fns"]
    #         os.remove(os.path.join(path, "pre_computed_img_feat_{}_rank_{}.pkl".format(split, rank)))
    #
    #     with open(os.path.join(path, "pre_computed_img_feat_{}.pkl".format(split)), 'wb') as f:
    #         pickle.dump(
    #             {
    #                 "img_feats": saved_embedding,
    #                 "num_bbs": saved_num_bbs,
    #                 "img_fns": data_loader.dataset.img_fns,
    #                 "metadata_fns": data_loader.dataset.metadata_fns
    #             },
    #             f,
    #             protocol=4)