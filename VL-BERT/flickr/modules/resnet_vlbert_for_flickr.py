import torch
import torch.nn as nn
from pretrain.modules.resnet_vlbert_for_pretraining_multitask import ResNetVLBERTForPretrainingMultitask


BERT_WEIGHTS_NAME = 'pytorch_model.bin'


class ResNetVLBERTforFlickr(ResNetVLBERTForPretrainingMultitask):
    def __init__(self, config):
        super(ResNetVLBERTforFlickr, self).__init__(config)
        self.itm_output = nn.Linear(config.NETWORK.VLBERT.hidden_size, 2)
        self.rank_output = nn.Linear(config.NETWORK.VLBERT.hidden_size, 1)
        self.init_output()

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self,
                image,
                boxes,
                im_info,
                text
                ):
        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=None,
                                                mask_visual_embed=None)

        ############################################

        # prepare text
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_token_type_ids = text.new_zeros(text.shape)
        text_mask = (text_input_ids > 0)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )

        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        relationship_logits, mlm_logits, mvrc_logits, text_out, object_out, pooled_rep = self.vlbert(text_input_ids,
                                                                   text_token_type_ids,
                                                                   text_visual_embeddings,
                                                                   text_mask,
                                                                   object_vl_embeddings,
                                                                   box_mask,
                                                                   output_all_heads=True)

        ###########################################
        outputs = {}
        rank_scores = self.rank_output(pooled_rep)

        outputs.update({
            'pooled_rep': pooled_rep,
            'rank_scores': rank_scores
        })

        return outputs, rank_scores

