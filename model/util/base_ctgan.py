import pandas as pd
from model.util.base import BaseSynthesizer
from model.util.need_function import *
from evaluate import compute_scores
from data import load_dataset

class BASIC_CTGANSynthesizer(BaseSynthesizer):
    def sampling_for_train(self, mean, std, data_sampler, G):
        fakez = torch.normal(mean=mean, std=std)

        condvec = self.cond_generator.sample(self.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = data_sampler.sample(self.batch_size, col, opt)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self.device)
            m1 = torch.from_numpy(m1).to(self.device)
            fakez = torch.cat([fakez, c1], dim=1)

            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
            c2 = c1[perm]

        fake = self.generator(fakez)
        fakeact = apply_activate(fake, self.transformer.output_info)

        real = torch.from_numpy(real.astype('float32')).to(self.device)

        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
        else:
            real_cat = real
            fake_cat = fake
        if not G:
            return real_cat, fake_cat
        else:
            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

            return fake_cat, cross_entropy

    def write_score(self, train_data, i, dataset_name, scores, best_score, writer, save_loc):
        syn_data = self.sample(train_data.shape[0])
        score = compute_scores(self.train, self.test, syn_data, self.meta)
        self.generator.train()
        tmp_score = set(list(score.columns)[1:]) - {"dataset", "iteration"}
        for idx in range(len(score)):
            for name_score in tmp_score:
                writer.add_scalar(score.loc[idx, "name"] + '/' + name_score, score.loc[idx, name_score], i)

        for name_score in tmp_score:
            writer.add_scalar(' average/' + name_score, score.loc[:, name_score].mean(), i)
            if name_score in ["macro_f1", "f1", "r2", "test_likelihood"]:
                avg_score = score.loc[:, name_score].mean()
        score[' epoch'] = i
        scores = pd.concat([scores, score])

        if avg_score > best_score:
            torch.save(self.generator.state_dict(),
                       save_loc + dataset_name + "/model/G_{}score_{}time.pth".format(avg_score, i))
            best_score = avg_score
        return scores, best_score

    def sample(self, n):
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())


        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)

    def fit_sample(self, train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns):
        self.fit(train_data, test_data, meta_data, dataset_name, categorical_columns, ordinal_columns)
        return self.sample(train_data.shape[0])

    def model_load(self, generator_location, dataset_name):
        train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(dataset_name,
                                                                                              benchmark=True)
        self.train = train_data.copy()
        self.test = test_data
        self.meta = meta_data
        self.transformer = self.transformer()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = self.Generator_class(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim).to(self.device)
        self.generator.load_state_dict(torch.load(generator_location, map_location = self.device))

    def model_test(self, times, dataset_name):
        train, test, meta, categoricals, ordinals = load_dataset(dataset_name, benchmark=True)
        lst = []
        for i in range(times):
            syn_data = self.sample(train.shape[0])
            tmp = compute_scores(train, test, syn_data, meta)
            tmp["iters"] = i
            lst.append(tmp)
        return pd.concat(lst, axis=0)

    def clipping_all_time(self,clamp_even):
        # clipping time points t.
        if clamp_even:
            total_num = len(self.all_time)
            with torch.no_grad():
                for j in range(total_num):
                    if j == 0:
                        start = 0 + 0.00001
                    else:
                        start = 1 / total_num * j + 0.00001

                    if j == len(self.all_time) - 1:
                        end = 1 - 0.00001
                    else:
                        end = 1 / total_num * (j + 1) - 0.00001
                    self.all_time[j] = self.all_time[j].clamp_(min=start, max=end)
        else:
            with torch.no_grad():
                for j in range(len(self.all_time)):
                    if j == 0:
                        start = 0 + 0.00001
                    else:
                        start = self.all_time[j - 1].item() + 0.00001

                    if j == len(self.all_time) - 1:
                        end = 1 - 0.00001
                    elif self.all_time[j + 1].item() - 0.00001 <= 0:
                        end = start + 0.00001
                    else:
                        end = self.all_time[j + 1].item() - 0.00001
                    self.all_time[j] = self.all_time[j].clamp_(min=start, max=end)
