import torch
import TransE_model

# 加载模型
model = torch.load("./TransE_emb.pth")
# 加载entity与relation的emb
ent_emb = model.ent_embeddings
rel_emb = model.rel_embeddings


# 加载GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 加载训练集
f_test = open("./FB15k/test2ID.txt", "r")
test_lines = f_test.readlines()
for i in range(len(test_lines)):
    test_lines[i] = test_lines[i].split()
    test_lines[i] = [int(j) for j in test_lines[i]]
test_triplets = torch.LongTensor(test_lines).to(device)


# 生成is_entity列表
id_entity = [[i, ent_emb.weight.data[i]] for i in range(len(ent_emb.weight.data))]


# 设置评估标准
hit_1 = 0.0
hit_10 = 0.0
mean_rank = 0.0


# 统计预测值
for triplet in test_triplets:
    head = ent_emb.weight.data[triplet[0]]
    relation = rel_emb.weight.data[triplet[1]]
    tail = ent_emb.weight.data[triplet[2]]
    dis_ls = []
    for i in range(len(ent_emb.weight.data)):
        dis_ls.append([i, torch.norm(head + relation, ent_emb.weight.data[i])])
