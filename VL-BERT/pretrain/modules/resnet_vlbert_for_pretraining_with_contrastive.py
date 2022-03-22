import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.visual_linguistic_bert import BaseModel
from external.pytorch_pretrained_bert.modeling import ACT2FN

from pretrain.modules.resnet_vlbert_for_pretraining import ResNetVLBERTForPretraining


class ResNetVLBERTForContrastivePretraining(ResNetVLBERTForPretraining):
    def __init__(self, config):
        super(ResNetVLBERTForContrastivePretraining, self).__init__(config)
        # self.num_gpus = len(config.GPUS.split(','))
        self.num_gpus = 1
        self.batch_size = config.TRAIN.BATCH_IMAGES * self.num_gpus

        self.register_buffer("temperature", torch.tensor(config.TRAIN.TEMPERATURE))
        self.register_buffer("negatives_mask",
                             (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float())

        self.contrastive_head = VlBertContrastiveHeadTransform(config.NETWORK.VLBERT)

        self.enable_contrastive_loss = config.NETWORK.WITH_CONTRASTIVE_LOSS

    #     def forward(self, *inputs):
    #         inputs1, inputs2 = list(zip(*[(tmp[:, 0], tmp[:,1]) for tmp in inputs]))
    #         (outputs1, loss1), (outputs2, loss2) = self.pre_forward(*inputs1), self.pre_forward(*inputs2)

    #         outputs = {}
    #         loss = 0
    #         if self.enable_contrastive_loss:
    #             z1, z2 = self.contrastive_head(outputs1['pooled_rep']), self.contrastive_head(outputs2['pooled_rep'])
    #             label1, label2 = outputs1['relationship_label'].contiguous(), outputs2['relationship_label'].contiguous()
    #             # contrastive_loss = self.pair_loss(z1, z2, label1, label2)

    #             """Distributed gathering of contrastive logits"""
    #             # gather_z1 = [z1.new_zeros(z1.shape) for _ in range(self.num_gpus)]
    #             # gather_z2 = [z2.new_zeros(z2.shape) for _ in range(self.num_gpus)]
    #             # torch.distributed.all_gather(gather_z1, z1)
    #             # torch.distributed.all_gather(gather_z2, z2)
    #             # gather_z1 = torch.cat(gather_z1, dim=0)
    #             # gather_z2 = torch.cat(gather_z2, dim=0)
    #             # gather_representation = torch.cat([gather_z1, gather_z2], dim=0)
    #             #
    #             # gather_label1 = [label1.new_zeros(label1.shape) for _ in range(self.num_gpus)]
    #             # gather_label2 = [label2.new_zeros(label2.shape) for _ in range(self.num_gpus)]
    #             # torch.distributed.all_gather(gather_label1, label1)
    #             # torch.distributed.all_gather(gather_label2, label2)
    #             # gather_label1 = torch.cat(gather_label1, dim=0)
    #             # gather_label2 = torch.cat(gather_label2, dim=0)
    #             # gather_index = torch.logical_or(gather_label1, gather_label2)

    #             # representation = torch.stack([z1, z2], dim=0)
    #             # gather_representation = [representation.new_zeros(representation.shape) for _ in range(self.num_gpus)]
    #             # torch.distributed.all_gather(gather_representation, representation)
    #             # gather_representation = torch.cat([torch.cat([representation[i] for representation in gather_representation], dim=0) for i in range(2)], dim=0)

    #             # index = torch.stack([label1, label2], dim=0)
    #             # gather_index = [index.new_zeros(index.shape) for _ in range(self.num_gpus)]
    #             # torch.distributed.all_gather(gather_index, index)
    #             # gather_index = [torch.cat([index[i] for index in gather_index], dim=0) for i in range(2)]
    #             # gather_index = torch.logical_or(gather_index[0], gather_index[1])

    #             gather_representation = torch.cat([z1, z2], dim=0)
    #             gather_index = torch.logical_or(label1, label2)
    #             contrastive_loss = self.pair_loss(gather_representation, gather_index)

    #             outputs.update({
    #                 "contrastive_logits": [z1, z2],
    #                 # "contrastive_index": torch.logical_or(label1, label2),
    #                 "contrastive_loss": contrastive_loss,
    #             })
    #             loss += outputs['contrastive_loss']

    #         outputs.update({
    #             'relationship_logits': torch.cat([outputs1['relationship_logits'], outputs2['relationship_logits']], dim=0) if self.config.NETWORK.WITH_REL_LOSS else None,
    #             'relationship_label': torch.cat([outputs1['relationship_label'], outputs2['relationship_label']], dim=0) if self.config.NETWORK.WITH_REL_LOSS else None,
    #             'mlm_logits': torch.cat([outputs1['mlm_logits'], outputs2['mlm_logits']], dim=0) if self.config.NETWORK.WITH_MLM_LOSS else None,
    #             'mlm_label': torch.cat([outputs1['mlm_label'], outputs2['mlm_label']], dim=0) if self.config.NETWORK.WITH_MLM_LOSS else None,
    #             'mvrc_logits': torch.cat([outputs1['mvrc_logits'], outputs2['mvrc_logits']], dim=0) if self.config.NETWORK.WITH_MVRC_LOSS else None,
    #             'mvrc_label': torch.cat([outputs1['mvrc_label'], outputs2['mvrc_label']], dim=0) if self.config.NETWORK.WITH_MVRC_LOSS else None,
    #             'relationship_loss': (outputs1['relationship_loss'] + outputs2['relationship_loss']),
    #             'mlm_loss': (outputs1['mlm_loss'] + outputs2['mlm_loss']),
    #             'mvrc_loss': (outputs1['mvrc_loss'] + outputs2['mvrc_loss']),
    #         })

    #         relationship_loss = F.cross_entropy(outputs['relationship_logits'], outputs['relationship_label'])
    #         mlm_loss = F.cross_entropy(outputs['mlm_logits'].view((-1, outputs['mlm_logits'].shape[-1])),
    #                                            outputs['mlm_label'].contiguous().view(-1),
    #                                            ignore_index=-1)
    #         loss = relationship_loss+mlm_loss
    #         # loss += outputs['mlm_loss'] + outputs['relationship_loss'] + outputs['mvrc_loss']
    #         return outputs, loss

    def forward(self, *inputs):
        (outputs, loss) = self.pre_forward(*inputs)
        if self.enable_contrastive_loss:
            length = outputs['pooled_rep'].shape[0] // 2

            representation = self.contrastive_head(outputs['pooled_rep'])
            label1, label2 = outputs['relationship_label'][:length].contiguous(), outputs['relationship_label'][
                                                                                  length:].contiguous()
            assert label1.sum() == label2.sum(), "label1 = label2"
            z1, z2 = representation[:length], representation[length:]
            # index = torch.logical_and(label1, label2)
            index = label1
            if index.sum() > 0:
                contrastive_loss = self.pair_loss(representation, index)

            else:
                contrastive_loss = representation.new_zeros(())

            loss += contrastive_loss
            outputs.update({
                "contrastive_logits": [z1, z2],
                "index": torch.logical_and(label1, label2),
                "contrastive_loss": contrastive_loss,
            })

        return outputs, loss

    def pre_forward(self,
                    image,
                    boxes,
                    im_info,
                    text,
                    relationship_label,
                    mlm_labels,
                    mvrc_ops,
                    mvrc_labels):
        outputs, loss = super(ResNetVLBERTForContrastivePretraining, self).forward(image,
                                                                                   boxes,
                                                                                   im_info,
                                                                                   text,
                                                                                   relationship_label,
                                                                                   mlm_labels,
                                                                                   mvrc_ops,
                                                                                   mvrc_labels)

        return outputs, loss

    # def pair_loss(self, z_i, z_j, label1, label2):
    #     """
    #     emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
    #     z_i, z_j as per SimCLR paper
    #     """
    #     label1 = label2 = torch.logical_or(label1, label2)
    #     index = torch.cat([label1, label2], dim=0)
    #     representations = torch.cat([z_i, z_j], dim=0)
    #     similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    #
    #     sim_ij = torch.diag(similarity_matrix, self.batch_size)
    #     sim_ji = torch.diag(similarity_matrix, -self.batch_size)
    #     positives = torch.cat([sim_ij, sim_ji], dim=0)[index]
    #
    #     nominator = torch.exp(positives / self.temperature)
    #     denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    #
    #     loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1)[index])
    #     loss = loss_partial.sum()
    #     return loss

    def pair_loss(self, representations, index):
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        positives = sim_ij[index]

        nominator = torch.exp(positives / self.temperature)
        denominator = (self.negatives_mask * torch.exp(similarity_matrix / self.temperature))[:self.batch_size]
        # print("nominator, ", nominator, "index, ", index, "torch.sum(denominator, dim=1)[index], ", torch.sum(denominator, dim=1)[index])
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1)[index])
        loss = loss_partial.mean()
        return loss


#     def pair_loss(self, representations):
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
#         # batch_size = int(self.batch_size//self.num_gpus)
#         positives = torch.diag(similarity_matrix, self.batch_size)

#         nominator = torch.exp(positives / self.temperature)
#         denominator = (self.negatives_mask * torch.exp(similarity_matrix / self.temperature))[:self.batch_size]

#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
#         loss = loss_partial.mean()
#         return loss


class VlBertContrastiveHeadTransform(BaseModel):
    def __init__(self, config, language_pretrained_model_path=None):
        super(VlBertContrastiveHeadTransform, self).__init__(config)

        self.dense = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size))
        self.apply(self.init_weights)
        if language_pretrained_model_path is not None:
            self.load_pretrained_model(language_pretrained_model_path)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.normalize(hidden_states, dim=1)
        return hidden_states

    def load_pretrained_model(self, language_pretrained_model_path):
        return None

