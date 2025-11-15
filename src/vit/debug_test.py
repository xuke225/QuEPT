import torch
import torch.nn as nn
a = torch.tensor(torch.arange(12).reshape(4,3), dtype=torch.float32)
b = torch.tensor(torch.arange(12, 24).reshape(4,3), dtype=torch.float32)
print(a)
print(b)
# ## similarity
# class CosineSimilarity(nn.Module):
 
#     def forward(self, tensor_1, tensor_2):
#         normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
#         normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
#         return (normalized_tensor_1 * normalized_tensor_2).sum(dim = -1)
    
# cosine_1 = nn.CosineSimilarity() # sample
# cosine_2 = nn.CosineSimilarity(dim= -1) # token_wise
# cosine_3 = CosineSimilarity()


# # similarity = cosine_1(a, b)
# # print(similarity)
# similarity = cosine_2(a, b)
# print(similarity)
# loss = 1 - similarity
# print(loss.mean())
x = a 
a = a @ b.t()
print(x)
print(a)
# similarity = cosine_3(a, b)
# print(similarity)


# _, idx = torch.topk(similarity, k=2)
# print(idx)
# for i in range(4):
#     if i in idx:
#         print(i)
# ## replace 
# e = []
# for i in range(a.size(0)):
#     if i % 2 == 0:
#         e.append(a[i])
#     else:
#         e.append(b[i])
# e = torch.cat([x for x in e])
# print(e)
# e = e.reshape(2,2,3)
# print(e)

