import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.visual_linguistic_bert import BaseModel
from external.pytorch_pretrained_bert.modeling import ACT2FN

from pretrain.modules.resnet_vlbert_for_pretraining import ResNetVLBERTForPretraining
from pretrain.modules.resnet_vlbert_for_pretraining_multitask import ResNetVLBERTForPretrainingMultitask

class VlBertEventHead(nn.Module):
    def __init__(self, config):
        super(VlBertEventHead, self).__init__()

        self.dense = nn.Linear(config.hidden_size, 4)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class VisualCometEventOrdering(ResNetVLBERTForPretraining):
    def __init__(self, config):
        super(VisualCometEventOrdering, self).__init__(config)
        self.event_head = VlBertEventHead(config.NETWORK.VLBERT)

    def forward(self, image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels
                ):

        outputs, loss = super(VisualCometEventOrdering, self).forward(
                image,
                boxes,
                im_info,
                text,
                None,
                mlm_labels,
                mvrc_ops,
                mvrc_labels)

        pooled_rep = outputs['pooled_rep']

        event_logits = self.event_head(outputs['pooled_rep'])

        event_loss = F.cross_entropy(event_logits, relationship_label, reduction='mean')
        outputs.update({
            'pooled_rep': pooled_rep,
            'event_logits': event_logits,
            'event_label': relationship_label,
            'event_loss': event_loss
        })
        return outputs, event_loss


class VlBertImageHead(nn.Module):
    def __init__(self, config):
        super(VlBertImageHead, self).__init__()

        self.diff = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

    def forward(self, x, y):
        ioc_scores = self.diff(x, y)
        return ioc_scores


class VisualCometImageOrdering(ResNetVLBERTForPretraining):
    def __init__(self, config):
        super(VisualCometImageOrdering, self).__init__(config)
        self.image_feat_mapping = nn.Linear(2048, config.NETWORK.VLBERT.visual_size)
        self.image_head = VlBertImageHead(config.NETWORK.VLBERT)

    def forward(self, image, boxes, im_info, text, ordering_label):
        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features = boxes[:, :, 4:]
            boxes[:, :, 4:] = box_features

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=None,
                                                mask_visual_embed=None)
        img_feats = obj_reps['obj_reps']

        # prepare text
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)
        text_visual_embeddings = self._collect_obj_reps(text_tags, img_feats)

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((img_feats, object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT
        relationship_logits, mlm_logits, mvrc_logits, text_out, object_out, pooled_rep = self.vlbert(text_input_ids,
                                                                                                     text_token_type_ids,
                                                                                                     text_visual_embeddings,
                                                                                                     text_mask,
                                                                                                     object_vl_embeddings,
                                                                                                     box_mask,
                                                                                                     output_all_heads=True)



        pooled_output = pooled_rep

        batch_size = pooled_output.shape[0] // 2
        z_i, z_j = pooled_output[:batch_size], pooled_output[batch_size:]
        ioc_scores = self.image_head(z_i, z_j)

        ioc_loss = F.binary_cross_entropy_with_logits(ioc_scores.squeeze(), ordering_label.half(), reduction='none')
        outputs = {}
        outputs.update({
            'pooled_rep': pooled_output,
            'image_scores': ioc_scores,
            'ordering_label': ordering_label,
            'ioc_loss': ioc_loss
        })
        return outputs, ioc_loss



