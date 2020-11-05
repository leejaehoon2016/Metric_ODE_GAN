from synthesizers._6_ctgan_odeD_num_various import CTGANSynthesizer as CT

short = ["adult","alarm", "asia", "child", "grid", "gridr", "insurance", "ring"]
mid = ["news", "mnist12", "census", "credit"]
long = ["covtype", "intrusion",  "mnist28"]

syn = CT()
syn.model_load("C:/load_model/G_adult_299_3.pth","adult")
print(syn.model_test(3,"adult"))
