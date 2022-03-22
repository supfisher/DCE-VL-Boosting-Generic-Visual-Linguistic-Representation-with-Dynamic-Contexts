"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict

import torch
from torch import nn
from .model import UniterPreTrainedModel, UniterModel, UniterConfig
import logging
logger = logging.getLogger(__name__)


class UniterForImageTextRetrieval(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        if compute_loss:
            # triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            return rank_loss
        else:
            return rank_scores

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            for key in state_dict.keys():
                new_key = None
                if 'module.vlbert.encoder' in key:
                    new_key = key.replace('module.vlbert.encoder', 'uniter.encoder')
                if 'module.vlbert.pooler' in key:
                    new_key = key.replace('module.vlbert.pooler', 'uniter.pooler')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
        else:
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                model.__class__.__name__,
                "\n\t".join(error_msgs)))
        return model


class UniterForImageTextRetrievalHardNeg(UniterForImageTextRetrieval):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config, img_dim, margin)
        self.hard_size = hard_size

    def forward(self, batch, sample_from='t', compute_loss=True):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].size(0)
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.size(0) == 1:
                batch['input_ids'] = input_ids.expand(batch_size, -1)
        elif sample_from == 'i':
            if img_feat.size(0) == 1:
                batch['img_feat'] = img_feat.expand(batch_size, -1, -1)
            if img_pos_feat.size(0) == 1:
                batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
        else:
            raise ValueError()

        if self.training and compute_loss:
            with torch.no_grad():
                self.eval()
                scores = super().forward(batch, compute_loss=False)
                hard_batch = self._get_hard_batch(batch, scores, sample_from)
                self.train()
            return super().forward(hard_batch, compute_loss=True)
        else:
            return super().forward(batch, compute_loss)

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size+1]

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]
            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['gather_index'] = gather_index

        return hard_batch

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        state_dict = state_dict['state_dict']
        for key in state_dict.keys():
            new_key = None
            if 'module.vlbert.encoder' in key:
                new_key = key.replace('module.vlbert.encoder', 'uniter.encoder')
            if 'module.vlbert.pooler' in key:
                new_key = key.replace('module.vlbert.pooler', 'uniter.pooler')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                model.__class__.__name__,
                "\n\t".join(error_msgs)))
        return model

