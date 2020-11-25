import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
z = z.mean()
print(z)

z.backward() #this calculate the gradient of z with respect to x dz/dx
print(x.grad) # the attribute grad stores the gradients

weights = torch.ones(4, requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad_zero_() #this must be done so the grad from previous epochs do not accumalate