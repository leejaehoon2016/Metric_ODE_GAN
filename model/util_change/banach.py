import torch
import torch.nn as nn
import torch.nn.functional as F

def write_num_denom(numer, denom, writer, i):
    val = {key: value.item() for key, value in zip(["numer", "denom"], [numer.mean(), denom.mean()])}
    writer.add_scalars('banach/num_denom', val, i)
    writer.add_scalar('banach/div', (numer / denom).mean(), i)


def banach_loss_original(d_real, d_generate, real, generate, writer, i, kind, lambda_banach, stability_regularizer_factor):
    if kind == "zero":
        return 0
    elif kind == "cosine":
        denom = 1 - F.cosine_similarity(real, generate, dim=1)
    elif kind == "l2":
        denom = torch.norm(real - generate, dim=1)
    else:
        assert 0
    numer = torch.abs(d_real - d_generate)
    d_regularizer_mean_stability = torch.mean(torch.square(d_real))
    write_num_denom(numer, denom, writer, i)
    return torch.mean((numer/denom - 1) ** 2) * lambda_banach + d_regularizer_mean_stability * stability_regularizer_factor

def banach_loss_ODE1(d_real, d_generate, MetricNet, ode_real, ode_fake, writer, i, kind, lambda_banach, stability_regularizer_factor):
    if kind == "zero":
        return 0
    elif kind == "metric":
        denom = MetricNet.test(ode_real, ode_fake)
    else:
        assert 0
    numer = torch.abs(d_real - d_generate)
    d_regularizer_mean_stability = torch.mean(torch.square(d_real))
    write_num_denom(numer, denom, writer, i)
    return torch.mean((numer/denom - 1) ** 2) * lambda_banach + d_regularizer_mean_stability * stability_regularizer_factor

def banach_loss_ODE_split(d_real, d_generate, MetricNet, real_cat, fake_cat, writer, i, kind, lambda_banach, stability_regularizer_factor):
    if kind == "zero":
        return 0
    elif kind == "metric":
        denom = MetricNet.test(real_cat, fake_cat)
    else:
        assert 0
    numer = torch.abs(d_real - d_generate)
    d_regularizer_mean_stability = torch.mean(torch.square(d_real))
    write_num_denom(numer, denom, writer, i)
    return torch.mean((numer/denom - 1) ** 2) * lambda_banach + d_regularizer_mean_stability * stability_regularizer_factor

