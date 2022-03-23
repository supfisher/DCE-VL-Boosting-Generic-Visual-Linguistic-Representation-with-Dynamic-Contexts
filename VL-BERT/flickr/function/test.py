import torch
import numpy as np
from flickr.data.build import make_dataloader
from flickr.data.transforms.build import build_transforms
from flickr.modules import *
from flickr.data.build import build_dataset
from flickr.data import samplers

from common.utils.load import smart_resume, smart_partial_load_model_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distributed
import random
from tqdm import tqdm
from flickr.data.collate_batch import BatchCollator
import pickle
import os, sys
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as Apex_DDP
except ImportError:
    pass

IMG_SIZE = 1000
TXT_SIZE = 1000 * 5


def to_cuda(batch):
    batch = list(batch)

    for i in range(len(batch)):
        if isinstance(batch[i], torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking=True)
        elif isinstance(batch[i], list):
            for j, o in enumerate(batch[i]):
                if isinstance(batch[i], torch.Tensor):
                    batch[i][j] = o.cuda(non_blocking=True)

    return batch


def reconstruct_txt_batch(batch, new_batch):
    for b, new_b in zip(batch, new_batch):
        b[3] = new_b[0]
        b[5] = new_b[1]
    return batch


def reconstruct_img_batch(batch, new_batch):
    for b, new_b in zip(batch, new_batch):
        b[0] = new_b[0]
        b[1] = new_b[1]
        b[2] = new_b[2]
        b[-1] = new_b[-1]
    return batch


def generated_model_loader(args, config):
    # manually set random seed
    if config.RNG_SEED > -1:
        random.seed(config.RNG_SEED)
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

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

        print(f'native distributed, size: {world_size}, local rank: {local_rank}')
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)

        model = model.cuda()
        if not config.TRAIN.FP16:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        test_dataloader = make_dataloader(config,
                                         mode='test',
                                         distributed=True,
                                         num_replicas=world_size,
                                         rank=rank)

    else:
        model = model.cuda()
        test_dataloader = make_dataloader(config, mode='test', distributed=False)

    if config.TRAIN.FP16:
        [model] = amp.initialize([model],
                                 opt_level='O2',
                                 keep_batchnorm_fp32=False)
        if args.dist:
            model = Apex_DDP(model, delay_allreduce=False)

    # partial load pretrain state dict
    if config.NETWORK.PARTIAL_PRETRAIN != "":
        pretrain_state_dict = \
            torch.load(config.NETWORK.PARTIAL_PRETRAIN, map_location=lambda storage, loc: storage)[
                'state_dict']
        prefix_change = [prefix_change.split('->') for prefix_change in
                         config.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES]
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
        distributed.barrier()
        for v in model.state_dict().values():
            distributed.broadcast(v, src=0)

    return test_dataloader, model


def generate_ITR_scores(test_dataloader, model):
    txt2img_scores = torch.ones([TXT_SIZE, IMG_SIZE]).cuda()*(-100)
    model.eval()

    for nbatch, batch in tqdm(enumerate(test_dataloader), 'ITR: '):
        img_index = batch[-2]
        txt_index = batch[-1]
        batch = to_cuda(batch[:-2])
        outputs, loss = model.forward(*batch)
        txt2img_scores[txt_index, img_index] = outputs['rank_scores'].view(-1).detach()

    return txt2img_scores


def test_net(args, config):
    if os.path.exists(os.path.join(args.model_dir, 'flickr_txt2img.pkt')):
        data = torch.load(os.path.join(args.model_dir, 'flickr_txt2img.pkt'))
        score_matrix = data['score_matrix']
    else:
        test_dataloader, model = generated_model_loader(args, config)
        score_matrix = generate_ITR_scores(test_dataloader, model)
        if args.dist:
            distributed.all_reduce(score_matrix)
        if not os.path.exists(args.model_dir):
         os.makedirs(args.model_dir)
        torch.save({'score_matrix': score_matrix},
                   os.path.join(args.model_dir, 'flickr_txt2img.pkt'))

    img2j = {i: j for j, i in enumerate(range(IMG_SIZE))}
    txt2i = {t: i for i, t in enumerate(range(TXT_SIZE))}
    txt2img = torch.tensor([[i]*5 for i in range(IMG_SIZE)]).view(-1).tolist()
    img2txts = torch.tensor([list(range(5*i, 5*i+5)) for i in range(IMG_SIZE)]).tolist()
    txt_ids = range(TXT_SIZE)
    img_ids = range(IMG_SIZE)
    # image retrieval
    _, rank_txt = score_matrix.topk(10, dim=1)
    gt_img_j = torch.LongTensor([img2j[txt2img[txt_id]]
                                 for txt_id in txt_ids],
                                ).to(rank_txt.device
                                     ).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == gt_img_j).nonzero()
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0

    # text retrieval
    _, rank_img = score_matrix.topk(10, dim=0)
    tr_r1, tr_r5, tr_r10 = 0, 0, 0
    for j, img_id in enumerate(img_ids):
        gt_is = [txt2i[t] for t in img2txts[img_id]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in gt_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_log = {'txt_r1': tr_r1,
                'txt_r5': tr_r5,
                'txt_r10': tr_r10,
                'txt_r_mean': tr_mean,
                'img_r1': ir_r1,
                'img_r5': ir_r5,
                'img_r10': ir_r10,
                'img_r_mean': ir_mean,
                'r_mean': r_mean}
    print(eval_log)
    return eval_log