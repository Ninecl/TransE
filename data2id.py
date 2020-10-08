import os


# 读取train，test，valid三个文件
print("load data......")
f_train = open("./FB15k/freebase_mtr100_mte100-train.txt", 'r')
f_test = open("./FB15k/freebase_mtr100_mte100-test.txt", 'r')
f_valid = open("./FB15k/freebase_mtr100_mte100-valid.txt", 'r')
lines_train = f_train.readlines()
lines_test = f_test.readlines()
lines_valid = f_valid.readlines()
print("load sucessfully!")

# 遍历train文件中的所有三元组，并做成(n, 3)的数组
train_triplets = []
for line in lines_train:
    train_triplets.append(line.split())
# 遍历test文件中的所有三元组，并做成(n, 3)的数组
test_triplets = []
for line in lines_test:
    test_triplets.append(line.split())
# 遍历valid文件中的所有三元组，并做成(n, 3)的数组
valid_triplets = []
for line in lines_valid:
    valid_triplets.append(line.split())

# 生成实体与关系的集合
entities = set()
relations = set()
for triplet in train_triplets:
    entities.add(triplet[0])
    entities.add(triplet[2])
    relations.add(triplet[1])
print("There are %d entities." % len(entities))
print("There are %d relations." % len(relations))

# 生成映射字典
entity2id = {}
relation2id = {}
entityID = 0
relationID = 0
# 遍历实体集合，建立映射字典
for entity in entities:
    entity2id[entity] = entityID
    entityID += 1
# 遍历关系集合，建立映射字典
for relation in relations:
    relation2id[relation] = relationID
    relationID += 1

# 生成entity2ID与relation2ID文件
f_entity2ID = open("./FB15k/entity2ID.txt", "w")
f_relation2ID = open("./FB15k/relation2ID.txt", "w")
# 遍历entity2ID字典，写文件
print("Writing entity2ID.txt......")
for k, v in entity2id.items():
    line = "{} {}\n".format(k, v)
    f_entity2ID.write(line)
# 遍历relation2ID字典，写文件
print("Writing realtion2ID.txt......")
for k, v in relation2id.items():
    line = "{} {}\n".format(k, v)
    f_relation2ID.write(line)

# 生成train、test、valid三个数据集映射到ID上的文件
f_train2ID = open("./FB15k/train2ID.txt", "w")
f_test2ID = open("./FB15k/test2ID.txt", "w")
f_valid2ID = open("./FB15k/valid2ID.txt", "w")
# 遍历train中的triplets，转化为ID
print("Writing train2ID.txt......")
for triplet in train_triplets:
    line = "{} {} {}\n".format(entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]])
    f_train2ID.write(line)
# 遍历test中的triplets，转化为ID
print("Writing test2ID.txt......")
for triplet in test_triplets:
    line = "{} {} {}\n".format(entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]])
    f_test2ID.write(line)
# 遍历valid中的triplets，转化为ID
print("Writing valid2ID.txt......")
for triplet in valid_triplets:
    line = "{} {} {}\n".format(entity2id[triplet[0]], relation2id[triplet[1]], entity2id[triplet[2]])
    f_valid2ID.write(line)

# 关闭文件
f_train.close()
f_test.close()
f_valid.close()
f_entity2ID.close()
f_relation2ID.close()
f_train2ID.close()
f_test2ID.close()
f_valid2ID.close()