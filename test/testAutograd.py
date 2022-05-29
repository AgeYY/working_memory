import torch

batch_size = 2
x = torch.randn((batch_size, 2), requires_grad=True)
x2 = torch.randn((batch_size, 2), requires_grad=True)
y = x * 2 + x2
z = y * 2
v = torch.ones_like(x)

print(x, y)

dydx, = torch.autograd.grad(z, x, grad_outputs=v, retain_graph=True, create_graph=True)

print(dydx.requires_grad)
