import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_distance(x1, x2, p):
    return 1 - F.cosine_similarity(x1, x2, dim=1)

class Net(nn.Module):
    def __init__(self,D_ode_dim):
        super(Net, self).__init__()
        self.dim = D_ode_dim
        self.lastdim = self.dim
        self.bn = nn.BatchNorm1d(self.dim)
        self.seq = nn.Sequential(
            nn.Linear(self.dim, 2 * self.dim),
            nn.ReLU(),
            nn.Linear(2 * self.dim, 2 * self.dim),
            nn.ReLU(),
            nn.Linear(2 * self.dim, self.dim)
        )
        self.last = nn.Sequential(nn.BatchNorm1d(self.dim, affine=False),
                                  nn.Linear(self.dim, self.lastdim, bias=False))

    def forward(self, x):
        output = self.bn(x)
        output = self.seq(output)
        output = output + x
        output = self.seq(output)
        output = output + x
        output = self.last(output)
        return output

class MetricNet_Basic(nn.Module):
    def __init__(self, D_ode_dim, dist_kind, margin, regularizer_lambda):
        super(MetricNet_Basic, self).__init__()
        self.embeddingnet = Net(D_ode_dim)
        self.margin = margin
        self.regularizer_lambda = regularizer_lambda
        if dist_kind == "cos":
            self.dist_func =  cosine_distance
            self.p = 0
        elif type(dist_kind) is int:
            self.dist_func = F.pairwise_distance
            self.p = dist_kind
        else:
            assert 0

    def train(self, true, fake, writer, i):
        embedded_true = self.embeddingnet(true)
        embedded_fake = self.embeddingnet(fake)
        perm_embedded_true = self.embeddingnet(true)[torch.randperm(embedded_fake.size()[0])]
        perm_embedded_fake = self.embeddingnet(fake)[torch.randperm(embedded_fake.size()[0])]

        dist_t_t = self.dist_func(embedded_true, perm_embedded_true, p = self.p)
        dist_t_f = self.dist_func(embedded_true, perm_embedded_fake, p = self.p)
        dist_f_f = self.dist_func(embedded_fake, perm_embedded_fake, p = self.p)
        dist_f_t = self.dist_func(embedded_fake, perm_embedded_true, p = self.p)
        val = {key: value.item() for key, value in zip(["dist_t_t", "dist_t_f", "dist_f_f"], [dist_t_t.mean(), dist_t_f.mean(), dist_f_f.mean()])}
        writer.add_scalars('banach/dist', val, i)

        regularizer = embedded_true.norm(2) + embedded_fake.norm(2)
        criterion = torch.nn.MarginRankingLoss(margin=self.margin)
        target_r = torch.ones_like(dist_t_t)
        loss_triplet_r = criterion(dist_t_f, dist_t_t, target_r)
        loss_triplet_f = criterion(dist_f_t, dist_f_f, target_r)
        total_triplet_loss = (loss_triplet_r + loss_triplet_f) / 2 + self.regularizer_lambda * regularizer

        return total_triplet_loss

    def test(self, true, fake):
        true, fake = torch.tensor(true), torch.tensor(fake)
        embedded_true = self.embeddingnet(true)
        embedded_fake = self.embeddingnet(fake)
        embedded_true = nn.functional.normalize(embedded_true, dim=1)
        embedded_fake = nn.functional.normalize(embedded_fake, dim=1)
        dist = self.dist_func(embedded_true, embedded_fake, p = self.p)
        return dist
