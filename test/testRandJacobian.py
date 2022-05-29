import context
from core.jacobian import JacobianReg as JReg
import torch

##### Test1 : get the derivertive of derivertive
#batch_size = 2
#x = torch.randn((batch_size, 2), requires_grad=True)
#
#y = x**2 + x2
#v = torch.ones_like(x)
#
#grad_x = torch.autograd.grad(y, x, v,
#                              retain_graph=True, 
#                              create_graph=True)
#
#print(grad_x.requires_grad)
#
##### Test2: try to use jreg
#jreg = JReg(n = -1)
#dydx = jreg(x, y)
#print(dydx.requires_grad)

#### Test3: grad in the loop

x = torch.randn((1, 2), requires_grad=True)
x_collector = [x]
v = torch.ones_like(x)

for i in range(5):
    x_new = x**2
    x = x_new + x
    x_collector.append(x)

for i in range(len(x_collector) - 1):
    grad_x = torch.autograd.grad(x_collector[i + 1], x_collector[i], v,
                              retain_graph=True, 
                              create_graph=True)
    print(grad_x, 2 * x_collector[i] + 1)

x = x.cuda()
x2 = x * 2
print(x2.device)
y = torch.randn(2)
print(x.device)
print(y.to(x.device))

