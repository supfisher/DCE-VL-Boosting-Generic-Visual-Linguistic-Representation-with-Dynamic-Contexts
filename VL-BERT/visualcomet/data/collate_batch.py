import torch
from common.utils.clip_pad import *
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip


def pad_tensors(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, hid, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def clip_pad_images(tensor, pad_shape, pad=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = torch.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = torch.arange(0, out_size, dtype=torch.long,
                                ).unsqueeze(0).repeat(batch_size, 1)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb,
                                                       dtype=torch.long).data
    return gather_index



class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.test_mode = self.dataset.test_mode
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        ordering_label = None # for image ordering classification
        len_batch = len(batch)
        if len(batch[0]) % 2 != 0:
            ordering_label = torch.tensor([i[-1] for i in batch])
            len_data = len(batch[0]) // 2
            for i in range(len_batch):
                batch.append(batch[i][len_data:len_data*2])
                batch[i] = batch[i][:len_data]

        if 'image' in self.data_names:
            if batch[0][self.data_names.index('image')] is not None:
                max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('image')].shape for data in batch]))
                image_none = False
            else:
                image_none = True
        if 'boxes' in self.data_names:
            max_boxes = max([data[self.data_names.index('boxes')].shape[0] for data in batch])
        if 'text' in self.data_names:
            max_text_length = max([len(data[self.data_names.index('text')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if 'image' in self.data_names:
                if image_none:
                    out['image'] = None
                else:
                    image = ibatch[self.data_names.index('image')]
                    out['image'] = clip_pad_images(image, max_shape, pad=0)

            if 'boxes' in self.data_names:
                boxes = ibatch[self.data_names.index('boxes')]
                out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            if 'text' in self.data_names:
                text = ibatch[self.data_names.index('text')]
                out['text'] = clip_pad_1d(text, max_text_length, pad=0)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            else:
                out_tuple += (torch.stack(tuple(items), dim=0),)

        if ordering_label is not None:
            out_tuple += (ordering_label, )

        return out_tuple
