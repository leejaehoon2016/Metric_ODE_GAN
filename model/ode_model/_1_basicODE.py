import torch as torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, first_layer_dim):
        super(ODEFunc, self).__init__()
        self.layer_start = nn.Sequential(nn.BatchNorm1d(first_layer_dim),
                                         nn.ReLU())

        self.layer_t = nn.Sequential(nn.Linear(first_layer_dim + 1, first_layer_dim * 2),
                                     nn.BatchNorm1d(first_layer_dim * 2),
                                     nn.Linear(first_layer_dim * 2, first_layer_dim * 1),
                                     nn.BatchNorm1d(first_layer_dim * 1))
        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out,tt],dim = 1)
        out = self.layer_t(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc, num_split, rtol, atol, device):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.num_split = num_split
        self.device = device
        self.rtol, self.atol = rtol, atol


    def forward(self, x):

        initial_value = x[0]
        integration_time = torch.cat(x[1], dim = 0).to(self.device)
        zero = torch.tensor([0.], requires_grad=False).to(self.device)
        one = torch.tensor([1.], requires_grad=False).to(self.device)
        all_time = torch.cat( [zero, integration_time, one],dim=0).to(self.device)
        self.total_integration_time1 = [all_time[i:i+2] for i in range(self.num_split)]

        out = [[1,initial_value]]
        for i in range(len(self.total_integration_time1)):
            self.integration_time = self.total_integration_time1[i].type_as(initial_value)
            out_ode = odeint(self.odefunc, out[i][1], self.integration_time, rtol = self.rtol, atol = self.atol)
            out.append(out_ode)
        return torch.cat([i[1] for i in out], dim=1)

def ODETime(num_split, device):
    return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device=device) for i in
            range(1, num_split)]

