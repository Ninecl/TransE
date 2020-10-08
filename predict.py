import torch


# 加载模型
model = torch.load("./TransE_emb.pth")
print(model.ent_embeddings.weight)