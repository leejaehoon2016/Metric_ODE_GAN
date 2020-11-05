from model.util.identity import IdentitySynthesizer as IS
from model._2_ctgan_ODE1_1 import CTGANSynthesizer as CT
from model._2_ctgan_ODE1_1 import save_loc
from benchmark import benchmark

import warnings, os
warnings.filterwarnings(action='ignore')

short = ["adult","alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
mid = ["news", "mnist12", "census", "credit"]
long = ["covtype", "intrusion",  "mnist28"]
DEFAULT_DATASETS = ["adult","alarm", "asia", "census", "child", "covtype", "credit", "grid", "gridr", "insurance",
                    "intrusion", "mnist12", "mnist28", "news", "ring"]

for kind in ["adult"]:
    print(kind)
    if not(os.path.isdir(save_loc[:-1])):
        os.makedirs(save_loc[:-1])
    if not (os.path.isdir(save_loc + kind)):
        os.makedirs(save_loc + kind)
    if not (os.path.isdir(save_loc + kind + "/" + "model")):
        os.makedirs(save_loc + kind + "/" + "model")
    if not (os.path.isdir(save_loc + kind + "/" + "score")):
        os.makedirs(save_loc + kind + "/" + "score")
    if not (os.path.isdir(save_loc + kind + "/" + "result")):
        os.makedirs(save_loc + kind + "/" + "result")

    a,b = benchmark(CT, datasets=[kind], iterations=1, add_leaderboard=False)
    a.to_csv(save_loc + kind + "/" + "result/result1.csv".format(kind))
    b.to_csv(save_loc + kind + "/" + "result/result2.csv".format(kind))
    # a, b = benchmark(IS, datasets=[kind], iterations=1, add_leaderboard=False)
    # a.to_csv(save_loc + "/{}_Identity1.csv".format(kind))
    # b.to_csv(save_loc + "/{}_Identity2.csv".format(kind))
    print()
    print()


