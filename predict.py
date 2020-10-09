import torch
import TransE_model
import time


# 加载模型
print("Loading model......")
model = torch.load("./TransE_emb.pth")
# 加载entity与relation的emb
ent_emb = model.ent_embeddings
rel_emb = model.rel_embeddings
print("Loading sucessfully!")


# 加载GPU
print("Loading GPU......")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Loading sucessfully!")


# 加载训练集
print("Loading test dataset......")
f_test = open("./FB15k/test2ID.txt", "r")
test_lines = f_test.readlines()
for i in range(len(test_lines)):
    test_lines[i] = test_lines[i].split()
    test_lines[i] = [int(j) for j in test_lines[i]]
test_triplets = torch.LongTensor(test_lines).to(device)
print("Loading sucessfully!")


# 设置评估标准与基础数据
hit_1 = 0.0
hit_10 = 0.0
mean_rank = 0.0
test_total = len(test_triplets)


# 统计预测值
print("Start to caculate Hit@1, Hit@10, MeanRank...")
cnt = 0
for triplet in test_triplets:
    # 确认head与relation的emb
    head_emb = ent_emb(triplet[0])
    relation_emb = rel_emb(triplet[1])
    # 确认tail的id
    tail_id = triplet[2]
    # 计算head+tail与所有实体的l2距离，并得到编号排序
    dis_ls = torch.norm(ent_emb.weight - (head_emb + relation_emb), p=2, dim=1)
    _, idx = torch.sort(dis_ls)
    # 生成idrank的排序字典
    id_dis_dict = dict(zip(idx.cpu().numpy(), range(len(idx))))
    # 计算hit@1
    if tail_id == idx[0]:
        hit_1 += 1
    if tail_id in idx[0: 10]:
        hit_10 += 1
    mean_rank += id_dis_dict[tail_id.item()]


# 输出评估结果
hit_1 /= test_total
hit_10 /= test_total
mean_rank /= test_total
print("Hit@1: {}".format(hit_1))
print("Hit@10: {}".format(hit_10))
print("MeanRank: {}".format(mean_rank))


# 将结果输出到文件中
f_test_result = open("./test_result.txt", "a")
f_test_result.write(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
f_test_result.write("Hit@1: {}\nHit@10: {}\nMeanRank: {}\n".format(hit_1, hit_10, mean_rank))