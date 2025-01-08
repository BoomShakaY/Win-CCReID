# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.pow(x, 2).sum(dim=1)).view(bs1, 1).repeat(1, bs2)) * \
                (torch.sqrt(torch.pow(y, 2).sum(dim=1).view(1, bs2).repeat(bs1, 1)))
    cosine = frac_up / frac_down
    cos_d = 1 - cosine
    return cos_d


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, normalize_feature=False, dist_style='cosine'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.dist_style = dist_style

    def forward(self, a_emb, a_pos, a_neg):
        if self.normalize_feature:
            a_emb = F.normalize(a_emb)
            a_pos = F.normalize(a_pos)
            a_neg = F.normalize(a_neg)

        # (8,1024) -> (8,8)
        if self.dist_style == 'cosine':
            pos_score = cosine_dist(a_emb, a_pos)
            neg_score = cosine_dist(a_emb, a_neg)
        elif self.dist_style == 'euclidean':
            pos_score = euclidean_dist(a_emb, a_pos)
            neg_score = euclidean_dist(a_emb, a_neg)
        else:
            print("The dist_style", self.dist_style, "is not available")
            raise RuntimeError

        y = torch.ones_like(pos_score)

        loss = self.margin_loss(neg_score, pos_score, y)

        return loss
