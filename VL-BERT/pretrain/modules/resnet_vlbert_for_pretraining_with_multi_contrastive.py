import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.visual_linguistic_bert import BaseModel
from external.pytorch_pretrained_bert.modeling import ACT2FN

from pretrain.modules.resnet_vlbert_for_pretraining_multitask import ResNetVLBERTForPretrainingMultitask


class ResNetVLBERTForMultiContrastivePretraining(ResNetVLBERTForPretrainingMultitask):
    def __init__(self, config):
        super(ResNetVLBERTForMultiContrastivePretraining, self).__init__(config)
        self.num_gpus = 1
        self.batch_size = config.TRAIN.BATCH_IMAGES[0] * self.num_gpus

        self.register_buffer("temperature", torch.tensor(config.TRAIN.TEMPERATURE))
        self.register_buffer("negatives_mask",
                             (~torch.eye(self.batch_size, self.batch_size, dtype=bool)).float())
        self.contrastive_head = VlBertContrastiveHeadTransform(config.NETWORK.VLBERT)

        self.enable_contrastive_loss = config.NETWORK.WITH_CONTRASTIVE_LOSS

    def forward(self, *inputs):
        (outputs, loss) = self.pre_forward(*inputs)

        if self.enable_contrastive_loss:
            contrastive_pooled_rep = outputs['pooled_rep'][:2*self.batch_size]
            length = self.batch_size

            representation = self.contrastive_head(contrastive_pooled_rep)
            label1, label2 = outputs['relationship_label'][:length].contiguous(), outputs['relationship_label'][
                                                                                  length:].contiguous()
            assert label1.sum() == label2.sum(), "label1 != label2, {} != {}".format(label1, label2)
            z1, z2 = representation[:length], representation[length:]
            # index = torch.logical_and(label1, label2)
            index = label1
            if index.sum() > 0:
                contrastive_loss, similarity_matrix = self.pair_loss(z1, z2, index)

            else:
                contrastive_loss = representation.new_zeros(())
                similarity_matrix = None

            loss += contrastive_loss
            outputs.update({
                "similarity_matrix": similarity_matrix,
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
                mvrc_labels,
                *aux):

        outputs, loss = super(ResNetVLBERTForMultiContrastivePretraining, self).forward(image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                *aux)
        return outputs, loss

    def pair_loss(self, z1, z2, index):

        similarity_matrix = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, 0)
        positives = sim_ij[index]

        nominator = torch.exp(positives / self.temperature)
        denominator = (self.negatives_mask * torch.exp(similarity_matrix / self.temperature))[:self.batch_size]
        # print("nominator, ", nominator, "index, ", index, "torch.sum(denominator, dim=1)[index], ", torch.sum(denominator, dim=1)[index])
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1)[index])
        loss = loss_partial.mean()
        return loss, similarity_matrix


class VlBertContrastiveHeadTransform(BaseModel):
    def __init__(self, config):
        super(VlBertContrastiveHeadTransform, self).__init__(config)

        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = F.normalize(hidden_states, dim=1)
        return hidden_states
