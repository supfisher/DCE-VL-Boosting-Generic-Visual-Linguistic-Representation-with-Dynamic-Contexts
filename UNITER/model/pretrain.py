"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead
from .model import UniterModel, UniterPreTrainedModel
from .ot import optimal_transport_dist


class RegionFeatureRegression(nn.Module):
    " for MRM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class ContrastiveHead(nn.Module):
    def __init__(self, hidden_size):
        super(ContrastiveHead, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            GELU(),
            LayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.normalize(hidden_states, dim=1)
        return hidden_states


class ImageOrdering(nn.Module):
    def __init__(self, hidden_size):
        super(ImageOrdering, self).__init__()
        # self.diff = nn.Sequential(
        #     nn.Linear(hidden_size*2, hidden_size),
        #     GELU(),
        #     nn.Linear(hidden_size, 1)
        # )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            GELU(),
            LayerNorm(hidden_size, eps=1e-12),
        )
        self.diff = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, x, y):
        # score = self.diff(torch.cat([x, y], dim=1))
        x, y = self.linear(x), self.linear(y)
        score = self.diff(x, y)
        return score


class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.con_head = ContrastiveHead(config.hidden_size)
        self.eoc_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            GELU(),
            LayerNorm(config.hidden_size, eps=1e-12),
            nn.Linear(config.hidden_size, 4)
        )
        self.ioc_head = ImageOrdering(config.hidden_size)
        self.apply(self.init_weights)
        self.mean_score = 0

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(input_ids, position_ids,
                                     img_feat, img_pos_feat,
                                     attention_mask, gather_index,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task.startswith('con'):
            return self.forward_con(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index, compute_loss)
        elif task.startswith('ood'):
            return self.forward_ood(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index, compute_loss)
        elif task.startswith('ioc'):
            ordering_label = batch['ordering']
            return self.forward_ioc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    ordering_label, compute_loss)
        elif task.startswith('eoc'):
            targets = batch['targets']
            return self.forward_eoc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, compute_loss)
        elif task.startswith('wocap'):
            ordering_label = batch['ordering']
            return self.forward_wocap(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    ordering_label, compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden).bool()
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad).to(txt_emb)
            ot_pos_dist = ot_dist.masked_select(targets == 1)
            ot_neg_dist = ot_dist.masked_select(targets == 0)
            ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label

    def forward_con(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, compute_loss=True):
        # img_feat = img_feat.new_zeros(img_feat.shape)
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      )
        pooled_output = self.uniter.pooler(sequence_output)
        representation = self.con_head(pooled_output)

        if compute_loss:
            contrastive_loss, _ = self.pair_loss(representation)
            return contrastive_loss
        else:
            contrastive_loss, similarity_matrix = self.pair_loss(representation)
            return contrastive_loss, similarity_matrix

    def pair_loss(self, representations):
        batch_size = representations.shape[0] // 2
        z_i, z_j = representations[:batch_size], representations[batch_size:]
        negatives_mask = (~torch.eye(z_i.shape[0], batch_size, dtype=bool)).float().to(z_i.device)
        temperature = 1.0
        similarity_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, 0)
        positives = sim_ij

        nominator = torch.exp(positives / temperature)
        denominator = (negatives_mask * torch.exp(similarity_matrix / temperature))[:batch_size]
        # print("nominator, ", nominator, "index, ", index, "torch.sum(denominator, dim=1)[index], ", torch.sum(denominator, dim=1)[index])
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        return loss_partial, similarity_matrix

    def forward_eoc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        vcc_scores = self.eoc_head(pooled_output)

        if compute_loss:
            vcc_loss = F.cross_entropy(vcc_scores, targets, reduction='none')
            return vcc_loss
        else:
            return vcc_scores

    def forward_ood(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, compute_loss):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        if compute_loss:
            pooled_output = self.con_head(pooled_output)
        else:
            pooled_output = F.normalize(pooled_output, dim=1)

        return pooled_output

    def forward_ioc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, ordering_label, compute_loss):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        pooled_output = self.uniter.pooler(sequence_output)
        batch_size = pooled_output.shape[0] // 2
        z_i, z_j = pooled_output[:batch_size], pooled_output[batch_size:]
        ioc_scores = self.ioc_head(z_i, z_j)

        if compute_loss:
            ioc_loss = F.binary_cross_entropy_with_logits(ioc_scores.squeeze(), ordering_label.half(), reduction='none')
            return ioc_loss
        else:
            return ioc_scores

    def forward_wocap(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, ordering_label, compute_loss):
        sequence_output = self.uniter(None, None,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        pooled_output = self.uniter.pooler(sequence_output)
        batch_size = pooled_output.shape[0] // 2
        z_i, z_j = pooled_output[:batch_size], pooled_output[batch_size:]
        ioc_scores = self.ioc_head(z_i, z_j)

        if compute_loss:
            ioc_loss = F.binary_cross_entropy_with_logits(ioc_scores.squeeze(), ordering_label.half(), reduction='none')
            return ioc_loss
        else:
            return ioc_scores

