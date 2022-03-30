import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SupInstanceDiscrimination(nn.Module):
    def __init__(self):
        super(SupInstanceDiscrimination, self).__init__()

    def forward(self, features, indices, labels):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        log_probs = F.log_softmax(features, dim=-1)

        mask = torch.zeros_like(log_probs)

        repeat_indices = indices.repeat(log_probs.size(0), 1)

        index_label_mat = (repeat_indices * label_mask).long()

        mask.scatter_(1, index_label_mat, 1)

        mask[:, 0] = 0

        mask_selected_mean_log_probs = log_probs * mask / torch.sum(mask, dim=1, keepdim=True)

        loss = - mask_selected_mean_log_probs
        loss = torch.sum(loss) / mask_selected_mean_log_probs.size(0)

        return loss