"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from .data import (TxtTokLmdb, DetectFeatLmdb,
                   ImageLmdbGroup, ConcatDatasetWithLens)
from .sampler import TokenBucketSampler
from .loader import PrefetchLoader, MetaLoader
from .vqa import VqaDataset, VqaEvalDataset, vqa_collate, vqa_eval_collate
from .ve import VeDataset, VeEvalDataset, ve_collate, ve_eval_collate
from .nlvr2 import (Nlvr2PairedDataset, Nlvr2PairedEvalDataset,
                    Nlvr2TripletDataset, Nlvr2TripletEvalDataset,
                    nlvr2_paired_collate, nlvr2_paired_eval_collate,
                    nlvr2_triplet_collate, nlvr2_triplet_eval_collate)
from .itm import (TokenBucketSamplerForItm, ItmDataset,
                  itm_collate, itm_ot_collate,
                  ItmRankDataset, ItmValDataset, ItmEvalDataset,
                  ItmRankDatasetHardNegFromImage,
                  ItmRankDatasetHardNegFromText,
                  itm_rank_collate, itm_val_collate, itm_eval_collate,
                  itm_rank_hn_collate)
from .mlm import MlmDataset, mlm_collate
from .mrm import MrfrDataset, MrcDataset, mrfr_collate, mrc_collate
from .vcr import (VcrTxtTokLmdb, VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate)
from .re import (ReTxtTokLmdb, ReDataset, ReEvalDataset,
                 re_collate, re_eval_collate)
from .pretrain_vcr import MlmDatasetForVCR, mlm_collate_for_vcr, MrfrDatasetForVCR, mrfr_collate_for_vcr, MrcDatasetForVCR, mrc_collate_for_vcr

from.contrastive import ConDataset, con_collate, EocDataset, OodDataset, ood_collate, IocDataset, ioc_collate
from.contrastive import IocDatasetWOCaption, ioc_collate_wo_cap
from.contrastive import OodVcrDataset
