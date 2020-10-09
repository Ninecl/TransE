import torch
import torch.optim as optim
import numpy as np
import time
import TransE_model
from tqdm import tqdm


EPOCHES = 100
BATCH_SIZE = 4000
LEARNING_RATE = 0.01
DIM = 100
MARGIN = 1.


# 加载GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载训练数据集，为ID形式
print("Loading triplets......")
f_train2ID = open("./FB15k/train2ID.txt", "r")
train_triplets = f_train2ID.readlines()
# 将字符串ID转化为数字
for i in range(0, len(train_triplets)):
    train_triplets[i] = train_triplets[i].split()
    for j in range(0, len(train_triplets[i])):
        train_triplets[i][j] = int(train_triplets[i][j])
# 创建训练数据集
train_triplets = torch.tensor(train_triplets).to(device)
print("Loading sucessfully.\nThere are %d triplets." % len(train_triplets))

# 加载所有实体
print("Loading entities......")
f_entity2ID = open("./FB15k/entity2ID.txt", "r")
entity2ID = f_entity2ID.readlines()
# 将字符串转化为数字
for i in range(0, len(entity2ID)):
    entity2ID[i] = entity2ID[i].split()
    entity2ID[i][1] = int(entity2ID[i][1])
print("Loading sucessfully. There are %d entities." % len(entity2ID))

# 加载所有关系
print("Loading relations......")
f_relation2ID = open("./FB15k/relation2ID.txt", "r")
relation2ID = f_relation2ID.readlines()
# 将字符串转化为数字
for i in range(0, len(relation2ID)):
    relation2ID[i] = relation2ID[i].split()
    relation2ID[i][1] = int(relation2ID[i][1])
print("Loading sucessfully. There are %d relations." % len(relation2ID))


# 加载模型
model = TransE_model.TransE(len(entity2ID), len(relation2ID), train_triplets, dim=DIM ,margin=MARGIN, device=device).to(device)
# 迭代训练
for epoch in range(EPOCHES):
    print("Epoch {}:\n".format(epoch))
    start_time = time.time()
    # 计算每个epoch的loss
    total_loss = 0
    # 每个epoch都除一次二范数
    model.ent_embeddings.weight.data = model.ent_embeddings.weight.data / torch.norm(model.ent_embeddings.weight.data, p=2)
    # 计算一共分多少个batch
    nbatch = int(len(train_triplets) / BATCH_SIZE) + 1
    # 设置优化器
    opt_SGD = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
    for i in tqdm(range(nbatch)):
        opt_SGD.zero_grad()
        t_batch = model.corrupt(BATCH_SIZE)
        loss = model.forward(t_batch)
        total_loss += loss
        loss.backward()
        opt_SGD.step()
    end_time = time.time()
    print("Epoch {}: loss = {}, cost {}(s).".format(epoch, total_loss, end_time - start_time))

PATH = './models/TransE_emb_{}.pkl'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
torch.save(model, PATH)
f_train_log = open("./models/train_log.txt", "a")
f_train_log.write(PATH + '\n')
f_train_log.write("epoch: {}, batchsize: {}, learning_rate: {}, dim: {}, margin: {}\n".format(EPOCHES, BATCH_SIZE, LEARNING_RATE, DIM, MARGIN))


