import context
import torch

def jacobian(W_rec, state):
    '''
    Jacobian for RNN with update rule: new_state = W_input * input + W_rec * act(state) + B + noise, where the act is softplus function
    W_rec (torch tensor [float] (n, n)): n is the size of the RNN
    state (torch tensor [float] (n))
    '''
    n = W_rec.size()[0]
    jacobian = torch.zeros((n, n))
    dh = 1 / (1 + torch.exp(-state))
    jacobian = W_rec.mm(dh)
    return torch.norm(jacobian)

n = 2
time_len = 1
batch_size = 2
W_rec = torch.eye(n)
state = torch.rand(time_len, batch_size, n)
state_dig = torch.diag_embed(state)

#print(state_dig.size(), W_rec.size())
#print(W_rec)


#### Method 1: expand w_rec then use matmul
W_rec_ext = torch.unsqueeze(W_rec, 0).repeat(batch_size ,1, 1)
W_rec_ext = torch.unsqueeze(W_rec_ext, 0).repeat(time_len ,1, 1, 1)
jacobian = torch.matmul(W_rec_ext, state_dig)

#for i in range(time_len):
#    jacobian[i] = torch.bmm(W_rec_ext[i], state_dig[i])
#    print(jacobian.size())

jacobian_flat = torch.flatten(jacobian)
jac = torch.linalg.norm(jacobian_flat, ord=2)

#### Method 2: for loop
jacobian = torch.zeros(W_rec_ext.size())
for i in range(time_len):
    for j in range(batch_size):
        state_dig = torch.diag(state[i, j, :])
        jacobian[i, j] = W_rec.mm(state_dig)
jacobian_flat = torch.flatten(jacobian)
jac2 = torch.linalg.norm(jacobian_flat, ord=2)

#### Comparing two methods
print(jac, jac2)

#jac = 0
#for i in range(time_len):
#    for j in range(batch_size):
#        jac = jac + jacobian(W_rec, state[i, j, :])
#jac = jac / time_len / batch_size
#print(jac)

#a = torch.randn(2, 2, 3)
#print(a)
#print(torch.diag_embed(a))
#
#import torch
#
#a = torch.rand(2, 3)
#print(a)
#b = torch.eye(a.size(1))
#c = a.unsqueeze(2).expand(*a.size(), a.size(1))
#d = c * b
#print(d)

