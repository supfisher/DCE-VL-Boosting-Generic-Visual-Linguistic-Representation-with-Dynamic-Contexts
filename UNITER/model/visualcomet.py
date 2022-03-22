from .pretrain import UniterForPretraining
from torch import nn
from .layer import BertOnlyMLMHead
from collections import defaultdict
from torch.nn import functional as F
import torch


class UniterVisualComet(UniterForPretraining):
    """ 2nd Stage Pretrain UNITER for VCR
    """
    def init_type_embedding(self):
        new_emb = nn.Embedding(4, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        for i in [0, 1]:
            emb = self.uniter.embeddings.token_type_embeddings.weight.data[i, :]
            new_emb.weight.data[i, :].copy_(emb)
        emb = self.uniter.embeddings.token_type_embeddings.weight.data[0, :]
        new_emb.weight.data[2, :].copy_(emb)
        new_emb.weight.data[3, :].copy_(emb)
        self.uniter.embeddings.token_type_embeddings = new_emb

    def init_word_embedding(self, num_special_tokens):
        orig_word_num = self.uniter.embeddings.word_embeddings.weight.size(0)
        new_emb = nn.Embedding(
            orig_word_num + num_special_tokens, self.uniter.config.hidden_size)
        new_emb.apply(self.init_weights)
        emb = self.uniter.embeddings.word_embeddings.weight.data
        new_emb.weight.data[:orig_word_num, :].copy_(emb)
        self.uniter.embeddings.word_embeddings = new_emb
        self.cls = BertOnlyMLMHead(
            self.uniter.config, self.uniter.embeddings.word_embeddings.weight)

    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(None, None,
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