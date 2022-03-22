import torch
from .eval_metric import EvalMetric

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


class ContrastiveLoss(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ContrastiveLoss, self).__init__('ContrasLoss', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            self.sum_metric += outputs['loss']
            self.num_inst += 1


class EventAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(EventAccuracy, self).__init__('EVEAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            logits = outputs['event_logits']
            label = outputs['event_label']
            self.sum_metric += float((logits.argmax(dim=1) == label).sum().item())
            self.num_inst += logits.shape[0]


class ImageAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ImageAccuracy, self).__init__('IOCAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            ioc_scores = outputs['image_scores'].squeeze()
            label = outputs['ordering_label']
            pred_labels = ioc_scores.new_zeros(ioc_scores.shape).int()
            pred_labels[ioc_scores > 0] = 1
            self.sum_metric += float((pred_labels == label).sum().item())
            self.num_inst += len(pred_labels)

class IOCLoss(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(IOCLoss, self).__init__('IOCLoss', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            ioc_loss = outputs['ioc_loss']
            self.sum_metric += float(ioc_loss.mean().item())
            self.num_inst += len(ioc_loss)


from polyleven import levenshtein
def label2ordering(labels, len_ordering):
    ordering = list(range(len_ordering))
    for i in range(len_ordering):
        "check how many 1s are for a specific index"
        positive = sum(labels[i * (len_ordering - 1): (i + 1) * (len_ordering - 1)]).item()
        ordering[i] = len_ordering - positive - 1
    return ordering


class ImageEdit(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(ImageEdit, self).__init__('IOCEdit', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            ioc_scores = outputs['image_scores']
            pred_labels = ioc_scores.new_zeros(ioc_scores.shape)
            pred_labels[ioc_scores > 0] = 1
            len_ordering = outputs['len_ordering']
            pred_ordering = label2ordering(pred_labels.squeeze().int(), len_ordering)

            self.sum_metric -= levenshtein(''.join([str(i) for i in range(len_ordering)]),
                                     ''.join([str(i) for i in pred_ordering]))
            self.num_inst += len_ordering