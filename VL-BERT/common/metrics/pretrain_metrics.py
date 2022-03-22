import torch
from .eval_metric import EvalMetric
import math
import torch.nn.functional as F

class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class RelationshipAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(RelationshipAccuracy, self).__init__('RelAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if outputs['relationship_logits'] is not None:
                logits = outputs['relationship_logits']
                label = outputs['relationship_label']
                self.sum_metric += float((logits.argmax(dim=1) == label).sum().item())
                self.num_inst += logits.shape[0]
            else:
                self.sum_metric += 0
                self.num_inst += 0


class MLMAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracy, self).__init__('MLMAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits']
            label = outputs['mlm_label']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyWVC(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyWVC, self).__init__('MLMAccWVC', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_wvc']
            label = outputs['mlm_label_wvc']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MLMAccuracyAUX(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MLMAccuracyAUX, self).__init__('MLMAccAUX', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mlm_logits_aux']
            label = outputs['mlm_label_aux']
            keep = (label != -1)
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep]).sum().item())
                self.num_inst += keep.sum().item()


class MVRCAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(MVRCAccuracy, self).__init__('MVRCAccuracy', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['mvrc_logits']
            label = outputs['mvrc_label']
            keep = (label.sum(2) - 1.0).abs() < 0.1
            if keep.sum() > 0:
                self.sum_metric += float((logits[keep].argmax(dim=1) == label[keep].argmax(dim=1)).sum().item())
                self.num_inst += keep.sum().item()


class ContrastiveSimilarity(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ContrastiveSimilarity, self).__init__('ContrastiveSimilarity', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if outputs['similarity_matrix'] is not None:
                similarity_matrix = outputs['similarity_matrix']
                pred_idx = similarity_matrix.argmax(dim=1)
                self.sum_metric += float((pred_idx == torch.arange(similarity_matrix.shape[0]).to(pred_idx.device)).sum().item())
                self.num_inst += similarity_matrix.shape[0]
            else:
                self.sum_metric += 0
                self.num_inst += 0


class ITAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ITAccuracy, self).__init__('ITAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            similarity_matrix = torch.arccos(
                F.cosine_similarity(outputs['text_embeddings'].unsqueeze(1), outputs['visual_embeddings'].unsqueeze(0), dim=2)) / math.pi
            negatives_mask = ~torch.eye(similarity_matrix.shape[0], similarity_matrix.shape[0], dtype=torch.bool, device=similarity_matrix.device)
            similarity_matrix, negatives_mask = similarity_matrix.flatten(), negatives_mask.flatten()

            similarity_matrix[similarity_matrix > 0.5] = 1
            similarity_matrix[similarity_matrix < 0.5] = 0
            self.sum_metric += float((similarity_matrix == negatives_mask).sum().item())
            self.num_inst += similarity_matrix.shape[0]


class ITRecall(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ITRecall, self).__init__('ITRecall', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = F.cosine_similarity(outputs['text_embeddings'], outputs['visual_embeddings'])
            label = outputs['label']
            logits[logits > 0] = 1
            logits[logits < 0] = 0
            self.sum_metric += float((logits == label).sum().item())
            self.num_inst += logits.shape[0]
