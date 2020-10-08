import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class TransE(nn.Module):
    """
    定义TransE模型
    """
    def __init__(self, entities_cnt, relations_cnt, triplets, dim = 100, p_norm = 1, margin = None, device = None):
        super(TransE, self).__init__()
        self.entities_cnt = entities_cnt    # 所有实体
        self.relations_cnt = relations_cnt  # 所有关系
        self.triplets = triplets    # 所有三元组
        self.dim = dim  # embedding维度
        self.margin = margin
        self.p_norm = p_norm
        self.device = device
        
        # 生成embedding字典
        self.ent_embeddings = nn.Embedding(self.entities_cnt, self.dim).to(self.device)
        self.rel_embeddings = nn.Embedding(self.relations_cnt, self.dim).to(self.device)

        # 初始化embedding
        emb_range = (6 / np.sqrt(dim))
        # 初始化relation
        nn.init.uniform(self.rel_embeddings.weight.data, -emb_range, emb_range)
        self.rel_embeddings.weight.data = self.rel_embeddings.weight.data / torch.norm(self.rel_embeddings.weight.data, p=2)
        # 初始化entity
        nn.init.uniform(self.ent_embeddings.weight.data, -emb_range, emb_range)
    
    def corrupt(self, b):
        # 设置t_batch
        t_batch = []
        # 取样s_batch
        s_batch = np.random.choice(range(len(self.triplets)), b, False)
        # 替换头或尾实体
        for triplet_ID in s_batch:
            # triplet = self.triplets[triplet_ID]..numpy()
            triplet = [self.triplets[triplet_ID][i].item() for i in range(3)]
            # 取0-1随机数，小于0.5替换头实体，大于0.5替换尾实体
            seed = random.random()
            corrupt_triplet = []
            if seed < 0.5:
                sample_head = random.randint(0, self.entities_cnt - 1)
                while sample_head == triplet[0]:
                    sample_head = random.randint(0, self.entities_cnt - 1)
                corrupt_triplet.append(sample_head)
                corrupt_triplet.extend(triplet[1: ])
            else:
                sample_tail = random.randint(0, self.entities_cnt - 1)
                while sample_tail == triplet[2]:
                    sample_tail = random.randint(0, self.entities_cnt - 1)
                corrupt_triplet.extend(triplet[0: 2])
                corrupt_triplet.append(sample_tail)
            t_batch.append([triplet, corrupt_triplet])
        # 返回生成的t_batch,为一个tensor
        return torch.LongTensor(t_batch).to(self.device)
    
    def forward(self, t_batch):
        # 设置loss值
        loss = 0.
        # 对t_batch内的每个三元组对，计算emb值
        for triplet, corrupt_triplet in t_batch:
            # 正确三元的三个embedding
            triplet_head = self.ent_embeddings(triplet[0])
            triplet_relation = self.rel_embeddings(triplet[1])
            triplet_tail = self.ent_embeddings(triplet[2])
            # 错误三元组的三个embedding
            corrupt_triplet_head = self.ent_embeddings(corrupt_triplet[0])
            corrupt_triplet_relation = self.rel_embeddings(corrupt_triplet[1])
            corrupt_triplet_tail = self.ent_embeddings(corrupt_triplet[2])
            # 计算正确三元组与错误三元组之间的距离
            d_triplet = torch.norm(triplet_head + triplet_relation - triplet_tail, p=2)
            d_corrupt_triplet = torch.norm(corrupt_triplet_head + corrupt_triplet_relation - corrupt_triplet_tail, p=2)
            l = self.margin + d_triplet - d_corrupt_triplet
            loss += F.relu(l)
        # 返回t_batch的误差    
        return loss



