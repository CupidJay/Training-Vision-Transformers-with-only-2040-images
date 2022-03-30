import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LSH(nn.Module):
    def __init__(self, input_dim, output_dim, std=1.0, with_l2=True, LSH_loss='BCE'):
        super(LSH, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std = std
        self.LSH_loss_type = LSH_loss

        self.LSH_weight = nn.Linear(self.input_dim, self.output_dim, bias=True)
        if with_l2:
            self.mse_loss = torch.nn.MSELoss(reduction='mean')
        else:
            self.mse_loss = None
        if LSH_loss == 'BCE':
            self.LSH_loss = nn.BCEWithLogitsLoss()
        elif LSH_loss == 'L2':
            self.LSH_loss = torch.nn.MSELoss(reduction='mean')
        elif LSH_loss == 'L1':
            self.LSH_loss = torch.nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError(LSH_loss)

        self._initialize()
        

    def _initialize(self):
        nn.init.normal_(self.LSH_weight.weight, mean=0.0, std=self.std)
        nn.init.constant_(self.LSH_weight.bias, 0)
        self.LSH_weight.weight.requires_grad_(False)
        self.LSH_weight.bias.requires_grad_(False)


    def init_bias(self, model_t, train_loader, print_freq=None, use_median=True):
        if use_median:
            print("=> Init LSH bias by median")
        else:
            print("=> Init LSH bias by mean")
        dataset_size = len(train_loader.dataset)
        if use_median:
            all_hash_value = torch.zeros(dataset_size, self.output_dim)
        else:
            mean = torch.zeros(self.output_dim)

        model_t.eval()

        for idx, data in enumerate(train_loader):
            input = data[0]

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()

            # ============= forward ==============
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=False)
                feat_t = [f.detach() for f in feat_t]
                hash_t = self.LSH_weight(feat_t[-1])

            if use_median:
                index = data[-1]
                all_hash_value[index] = hash_t.cpu()
            else:
                mean += hash_t.sum(0).cpu() / dataset_size
            if print_freq is not None:
                if idx % print_freq == 0:
                    print("Init Bias: [{}/{}]".format(idx, len(train_loader)))

        if use_median:
            self.LSH_weight.bias.data[:] = - all_hash_value.median(0)[0]
        else:
            self.LSH_weight.bias.data[:] = - mean


    def forward(self, f_s, f_t):
        if self.mse_loss:
            l2_loss = self.mse_loss(f_s, f_t)
        else:
            l2_loss = 0
        hash_s = self.LSH_weight(f_s)
        hash_t = self.LSH_weight(f_t)
        if self.LSH_loss_type == 'BCE':
            pseudo_label = (hash_t > 0).float()
            loss = self.LSH_loss(hash_s, pseudo_label) 
        else:
            loss = self.LSH_loss(hash_s, hash_t)
        return l2_loss + loss

