from model.util.utils import BGMTransformer as Transfromer # minmax로 바꾸기 GeneralTransformer
from model.util_change.banach import banach_loss_original
from model.util_change.gradient_penalty import calc_gradient_penalty_original

# Initialize
GPU_NUM = 3 # 원하는 GPU
random_num = 777 # Reproduce

# GAN Model Hyper
lambda_grad = 1.0  # 이거원래는 10임(1.0)
lambda_banach = 0.1
stability_regularizer_factor = 1e-5
kind = "zero" # ["zero", "cosine", "l2"]

embedding_dim = 128 # z vector size
gen_dim = (256, 256)
dis_dim = (256, 256)
PACK = 1
l2scale = 1e-6 # G l2 scale
batch_size = 3000
epochs = 600
D_Learning_iter = 3

# optimizer
G_lr, D_lr= 2e-4, 2e-4
G_beta, D_beta = (0.5,0.9), (0.5,0.9)

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
    def __init__(self, input_dim, dis_dims, pack = PACK):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))

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
        discriminator = Discriminator(data_dim + self.cond_generator.n_opt, self.dis_dim).to(self.device)

        optimizerG = optim.Adam(self.generator.parameters(), lr = G_lr, betas = G_beta, weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr = D_lr, betas = D_beta)

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device = self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size
        for i in range(self.epochs):
            for id_ in range(steps_per_epoch):
                for _ in range(D_Learning_iter):
                    ########################
                    # Update D
                    ########################
                    real_cat, fake_cat = self.sampling_for_train(mean, std, data_sampler, G=False)

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    pen = calc_gradient_penalty_original(discriminator, real_cat, fake_cat, PACK, self.device, lambda_grad)
                    banach = banach_loss_original(y_real, y_fake, real_cat, fake_cat, writer, i, kind, lambda_banach, stability_regularizer_factor)
                    loss_d = loss_d + pen + banach
                    writer.add_scalar('losses/D_loss', (loss_d), i)

                    optimizerD.zero_grad()
                    loss_d.backward()
                    optimizerD.step()

                ########################
                # Update G
                ########################
                fake_cat, cross_entropy = self.sampling_for_train(mean, std, data_sampler, G=True)
                y_fake = discriminator(fake_cat)

                loss_g = -torch.mean(y_fake) + cross_entropy
                writer.add_scalar('losses/G_loss', loss_g, i)

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            if (i+1) % 10 == 0:
                print(i + 1 ,end = " ")
            scores, best_score = self.write_score(train_data, i, dataset_name, scores, best_score, writer, save_loc)
        scores.to_csv(save_loc + dataset_name + "/score/scores.csv")