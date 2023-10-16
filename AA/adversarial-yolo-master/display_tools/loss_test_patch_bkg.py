import torch
import torch.nn.functional as F

def tensor_correlation(tensor1, tensor2):
    # 将张量移动到 GPU 上
    tensor1 = tensor1.cuda()
    tensor2 = tensor2.cuda()

    # 将张量转换为浮点型
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()

    # 计算皮尔逊相关系数
    pearson_corr = torch.nn.functional.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)

    # 计算余弦相似度
    cosine_sim = F.cosine_similarity(tensor1.flatten(), tensor2.flatten(), dim=0)

    # 计算欧氏距离
    euclidean_dist = torch.dist(tensor1.flatten(), tensor2.flatten())

    # 求平均
    avg_corr = (pearson_corr + cosine_sim + euclidean_dist) / 3

    return avg_corr

# 创建两个示例张量
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

# 将张量移动到 GPU 上
tensor1 = tensor1.cuda()
tensor2 = tensor2.cuda()

# 计算相关性度量并求平均
avg_correlation = tensor_correlation(tensor1, tensor2)
print(avg_correlation)