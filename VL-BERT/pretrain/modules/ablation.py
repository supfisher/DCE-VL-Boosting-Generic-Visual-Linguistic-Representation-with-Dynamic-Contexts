import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.visual_linguistic_bert import BaseModel
from external.pytorch_pretrained_bert.modeling import ACT2FN

from pretrain.modules.resnet_vlbert_for_pretraining_multitask import ResNetVLBERTForPretrainingMultitask
from pretrain.modules.resnet_vlbert_for_pretraining import ResNetVLBERTForPretraining


class ResNetVLBERTForAblationPretrainingMultiTask(ResNetVLBERTForPretrainingMultitask):
    def __init__(self, config):
        super(ResNetVLBERTForAblationPretrainingMultiTask, self).__init__(config)

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                *aux):

        outputs, loss = super(ResNetVLBERTForAblationPretrainingMultiTask, self).forward(image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                *aux)
        return outputs, loss


class ResNetVLBERTForAblationPretraining(ResNetVLBERTForPretraining):
    def __init__(self, config):
        super(ResNetVLBERTForAblationPretraining, self).__init__(config)

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels):

        outputs, loss = super(ResNetVLBERTForAblationPretraining, self).forward(image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels)
        return outputs, loss
