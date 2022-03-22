import argparse
import subprocess

from visualcomet.function.config import config, update_config
import os
import pprint
import random

import numpy as np
import torch
import torch.nn
import torch.distributed as distributed
from common.utils.load import smart_partial_load_model_state_dict
from common.trainer import to_cuda
from visualcomet.data.build import make_dataloader, make_dataloaders
from visualcomet.modules import *
from visualcomet.function.val import do_validation
from tqdm import tqdm
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex if you want to use fp16.")


def parse_args():
    parser = argparse.ArgumentParser('Visualize pooled_rep')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--slurm', help='whether this is a slurm job', default=False, action='store_true')
    parser.add_argument('--model-dir', help='directory to save pooled_rep', type=str)

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    if args.slurm:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)

    return args, config


def val_net(args, config):
    # manually set random seed
    if config.RNG_SEED > -1:
        random.seed(config.RNG_SEED)
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    model = eval(config.MODULE)(config)
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

    model.cuda()
    train_loader = make_dataloader(config, mode='train', distributed=args.dist)
    val_loader = make_dataloader(config, mode='val', distributed=args.dist)

    # partial load pretrain state dict
    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)['state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
        if len(prefix_change) > 0:
            pretrain_state_dict_parsed = {}
            for k, v in pretrain_state_dict.items():
                no_match = True
                for pretrain_prefix, new_prefix in prefix_change:
                    if k.startswith(pretrain_prefix):
                        k = new_prefix + k[len(pretrain_prefix):]
                        pretrain_state_dict_parsed[k] = v
                        no_match = False
                        break
                if no_match:
                    pretrain_state_dict_parsed[k] = v
            pretrain_state_dict = pretrain_state_dict_parsed
        smart_partial_load_model_state_dict(model, pretrain_state_dict)

    if args.dist:
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)

    [model] = amp.initialize([model],
                             opt_level='O2',
                             keep_batchnorm_fp32=False)
    img_fn, pooled_rep = do_validation(model, train_loader)
    torch.save({"pooled_rep": torch.cat(pooled_rep, dim=0), "img_fname": img_fn},
               os.path.join(args.model_dir, "pooled_reps_{}.pkl".format(local_rank)))


@torch.no_grad()
def do_validation(net, val_loader):
    net.eval()
    img_fn = []
    pooled_rep = []
    for nbatch, batch in tqdm(enumerate(val_loader), 'Validation: '):
        img_fn.extend(batch[-1])
        batch = batch[:-1]
        batch = to_cuda(batch)
        outputs, _ = net(*batch)
        pooled_rep.extend(outputs['pooled_rep'])
    return img_fn, pooled_rep


def main():
    args, config = parse_args()
    val_net(args, config)


if __name__ == '__main__':
    main()