from collections import namedtuple
import torch
from common.trainer import to_cuda
from visualcomet.data.collate_batch import BatchCollator

# @torch.no_grad()
# def do_validation(net, val_loader, metrics, label_index_in_batch):
#     net.eval()
#     metrics.reset()
#     for nbatch, batch in enumerate(val_loader):
#         batch = to_cuda(batch)
#         outputs, _ = net(*batch)
#         metrics.update(outputs)


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    net.eval()
    metrics.reset()
    dataset = val_loader.dataset
    ioc_collate = BatchCollator(dataset)

    for i, samples in enumerate(dataset.image_ordering_one_cluster):
        inputs, ordering = dataset.get_one_cluster(samples)
        batch = ioc_collate(list(inputs))
        batch = to_cuda(batch)
        outputs, ioc_loss = net(*batch)
        outputs['len_ordering'] = len(ordering)
        metrics.update(outputs)

