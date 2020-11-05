from model.util.utils import BGMTransformer as Transfromer # minmax로 바꾸기 GeneralTransformer
from model.util_change.banach import banach_loss_ODE1
from model.util_change.gradient_penalty import calc_gradient_penalty_ODE1
from model.ode_model._1_basicODE import *
from model.metric_model._2_proxy import MetricNet_Proxy
# Initialize
GPU_NUM = 3 # 원하는 GPU
random_num = 777 # Reproduce

## ODE
rtol=1e-3
atol=1e-3
num_split = 3
clamp_even = False

# GAN Model Hyper
lambda_grad = 1.0  # 이거원래는 10임(1.0)
lambda_banach = 0.1
stability_regularizer_factor = 1e-5
kind = "metric" # ["zero", "metric"]

embedding_dim = 128 # z vector size
gen_dim = (256, 256)
dis_dim = (256, 256)
PACK = 1
l2scale = 1e-6 # G l2 scale
batch_size = 1000
epochs = 600
D_Learning_iter = 3
Metric_Learning_iter = 3

# optimizer
G_lr, D_lr, T_lr, M_lr = 2e-4, 2e-4, 2e-4, 2e-4
G_beta, D_beta, T_beta, M_beta = (0.5,0.9), (0.5,0.9), (0.5,0.9), (0.5,0.9)
T_step_size, M_step_size = 30, 30
T_gamma, M_gamma = 0.99, 0.99

# Metric
temperature = 0.05


# save location
save_loc = "/home/ljh5694/CTGAN/original/"

########################
# IMPORT
########################
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch.utils.tensorboard import SummaryWriter
from model.util.base_ctgan import BASIC_CTGANSynthesizer
from model.util.need_function import Sampler, Cond


import pandas as pd
import warnings, random
warnings.filterwarnings(action='ignore')
writer = SummaryWriter()

device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

random.seed(random_num)
torch.manual_seed(random_num)
torch.cuda.manual_seed_all(random_num)


########################
# GD MODEL Part
########################
class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, num_split, pack=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        self.num_split = num_split
        seq = []
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.ode = ODEBlock(ODEFunc(dim),num_split, rtol, atol, device)
        if num_split == 2:
            self.last = Sequential(nn.Linear(dim * (num_split + 1), dim * ((num_split + 1) // 2)),
                                   nn.Linear(dim * ((num_split + 1) // 2), 1))
        else:
            self.last = Sequential(nn.Linear(dim * (num_split + 1), dim * ((num_split + 1) // 2)),
                                   nn.Linear(dim * ((num_split + 1) // 2), dim * ((num_split + 1) // 2 // 2)),
                                   nn.Linear(dim * ((num_split + 1) // 2 // 2), 1))

    def forward(self, x):
        value = x[0]
        time = x[1]
        assert value.size()[0] % self.pack == 0
        out = self.seq(value.view(-1, self.packdim))
        out1_time = [out, time]
        out_ode = self.ode(out1_time)
        out = self.last(out_ode)
        return out, out_ode

class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)

class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [
                Residual(dim, item)
            ]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data

#######################
# Fitting Part
#######################
class CTGANSynthesizer(BASIC_CTGANSynthesizer):

    def __init__(self, embedding_dim = embedding_dim, gen_dim = gen_dim, dis_dim = dis_dim,
                 l2scale = l2scale, batch_size = batch_size, epochs = epochs, transformer = Transfromer):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.transformer = transformer
        self.Generator_class = Generator

    def fit(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        ## Needed for Scoring
        best_score = -100
        scores = pd.DataFrame()

        ## Model Make
        self.train = train_data.copy()
        self.test, self.meta = test_data, meta_data
        self.transformer = self.transformer(random_num)
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)

        train_data = self.transformer.transform(train_data)
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim

        self.cond_generator = Cond(train_data, self.transformer.output_info)
        self.generator = self.Generator_class(self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim).to(self.device)
        discriminator = Discriminator(data_dim + self.cond_generator.n_opt, self.dis_dim, num_split).to(self.device)
        self.all_time = ODETime(num_split,device)
        Mnet = MetricNet_Proxy(self.dis_dim[-1] * (num_split + 1), batch_size, temperature).cuda()

        optimizerG = optim.Adam(self.generator.parameters(), lr = G_lr, betas = G_beta, weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr = D_lr, betas = D_beta)
        optimizerT = torch.optim.Adam(self.all_time, lr = T_lr, betas = T_beta)
        optimizerM = torch.optim.Adam(Mnet.parameters(), lr=M_lr, betas = M_beta)
        optT_scheduler = optim.lr_scheduler.StepLR(optimizerT, step_size = T_step_size, gamma = T_gamma)
        optM_scheduler = optim.lr_scheduler.StepLR(optimizerM, step_size = M_step_size, gamma = M_gamma)

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device = self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):
                for _ in range(D_Learning_iter):
                    # ####################
                    # Update Metric
                    # ####################
                    for _ in range(Metric_Learning_iter):
                        real_cat, fake_cat = self.sampling_for_train(mean, std, data_sampler, G = False)
                        y_fake, ode_fake = discriminator([fake_cat, self.all_time])
                        y_real, ode_real = discriminator([real_cat, self.all_time])
                        Metric_loss = Mnet.train(ode_real, ode_fake, writer, i)
                        writer.add_scalar('losses/M_loss', Metric_loss, i)

                        optimizerM.zero_grad()
                        optimizerT.zero_grad()
                        with torch.autograd.set_detect_anomaly(True):
                            Metric_loss.backward()
                        optimizerM.step()
                        optimizerT.step()
                        optM_scheduler.step()
                        optT_scheduler.step()
                        self.clipping_all_time(clamp_even)
                        times = {'t' + str(t + 1): time.item() for t, time in enumerate(self.all_time)}
                        writer.add_scalars('time_points', times, i)

                    ########################
                    # Update D
                    ########################
                    real_cat, fake_cat = self.sampling_for_train(mean, std, data_sampler, G=False)

                    y_fake, ode_fake = discriminator([fake_cat, self.all_time])
                    y_real, ode_real = discriminator([real_cat, self.all_time])

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    pen = calc_gradient_penalty_ODE1(discriminator, real_cat, fake_cat, self.all_time, PACK, self.device, lambda_grad)
                    banach = banach_loss_ODE1(y_real, y_fake, Mnet, ode_real, ode_fake, writer, i, kind,
                                              lambda_banach, stability_regularizer_factor)
                    loss_d = loss_d + pen + banach
                    writer.add_scalar('losses/D_loss', (loss_d), i)

                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()

                ########################
                # Update G
                ########################
                fake_cat, cross_entropy = self.sampling_for_train(mean, std, data_sampler, G=True)
                y_fake, _ = discriminator([fake_cat,self.all_time])

                loss_g = -torch.mean(y_fake) + cross_entropy
                writer.add_scalar('losses/G_loss', loss_g, i)

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            if (i+1) % 10 == 0:
                print(i + 1 ,end = " ")
            scores, best_score = self.write_score(train_data, i, dataset_name, scores, best_score, writer, save_loc)
        scores.to_csv(save_loc + dataset_name + "/score/scores.csv")