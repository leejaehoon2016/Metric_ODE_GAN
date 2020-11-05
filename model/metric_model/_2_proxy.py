import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NormSoftmaxLoss(nn.Module):
    def __init__(self, dim, temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()
        num_instances = 2
        self.weight = nn.Parameter(torch.Tensor(num_instances, dim),requires_grad = True)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)
        prediction_logits = nn.functional.linear(embeddings, norm_weight)
        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets.cuda())
        return loss

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

class MetricNet_Proxy(nn.Module):
    def __init__(self, D_ode_dim, batch_size, temperature):
        super(MetricNet_Proxy, self).__init__()
        self.embeddingnet = Net(D_ode_dim)
        self.loss = NormSoftmaxLoss(self.embeddingnet.lastdim, temperature = temperature)
        self.target_true, self.target_fake = torch.ones((batch_size,)), torch.zeros((batch_size,))

    def train(self, true, fake, writer, i):
        embedded_true = self.embeddingnet(true)
        embedded_fake = self.embeddingnet(fake)
        embedded_true = nn.functional.normalize(embedded_true, dim=1)
        embedded_fake = nn.functional.normalize(embedded_fake, dim=1)
        perm = torch.randperm(embedded_fake.size()[0])
        dist_t_t = 1 - F.cosine_similarity(embedded_true, embedded_true[perm], dim=1)
        dist_t_f = 1 - F.cosine_similarity(embedded_true, embedded_fake[perm], dim=1)
        dist_f_f = 1 - F.cosine_similarity(embedded_fake, embedded_fake[perm], dim=1)
        val = {key: value.item() for key, value in zip(["dist_t_t", "dist_t_f", "dist_f_f"], [dist_t_t.mean(), dist_t_f.mean(), dist_f_f.mean()])}
        writer.add_scalars('banach/dist', val, i)
        proxy_dist = 1 - F.cosine_similarity(self.loss.weight[[0]], self.loss.weight[[1]])
        writer.add_scalar('banach/proxy_dist', proxy_dist, i)
        loss = self.loss(torch.cat([embedded_true, embedded_fake], dim = 0), torch.cat([self.target_true, self.target_fake], dim = 0).long())
        return loss

    def test(self, true, fake):
        true, fake = torch.tensor(true), torch.tensor(fake)
        # embedded_true = self.embeddingnet(true)
        embedded_fake = self.embeddingnet(fake)
        # embedded_true = nn.functional.normalize(embedded_true, dim=1)
        embedded_fake = nn.functional.normalize(embedded_fake, dim=1)
        dist = 1 - F.cosine_similarity(self.loss.weight[[1]], embedded_fake,dim = 1)
        return dist
