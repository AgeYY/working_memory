import torch

#tensor1 = torch.randn(4)
#tensor2 = torch.randn(4, 3)
tensor1 = torch.tensor([1, 2])
tensor2 = torch.tensor([[1, 0], [2, 1]])
ans = torch.matmul(tensor1, tensor2)
print(ans)
