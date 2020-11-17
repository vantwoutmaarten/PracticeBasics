import torch

#==================================================#
#               initializing Tensor                #
#==================================================#

#requires grad is needed for autograd
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

# Other common initialization methods
x = torch.empty(size = (3,3))
x = torch.zeros((3,3))
x = torch.rand((3,3))
x = torch.ones((3,3))
x = torch.eye(5,5) # I is the identity matrix
x = torch.arange(start = 0, end = 5, step =1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1)
x = torch.diag(torch.ones(3))
print(x)

# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short()) #create int16
print(tensor.long()) #create int64
print(tensor.half()) #float16 for new gpu
print(tensor.float()) #float32 (important)

#Array to tensor conversion and vice-versa
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()

#====================================================================#
#               Tensor Math and Comparison Operations                #
#====================================================================#

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])   

# Addition
z1 = torch.add(x,y)
z2 = x+y
# Subtraction
z = x-y

# Division
z = torch.true_divide(x,y) # element-wise division if equal size

# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x.pow(2)
z = x**2

# Simple comparison 
z = x > 0
print(z)

# Matrix multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = x1.mm(x2) # 2x3

# matrix exponentiation
matrix_exp = torch.rand((5,5))
matrix_exp = matrix_exp.matrix_power(3)

# element wise multiplication
z = x * y
print(z)

# dot product
z = torch.dot(x,y)

# Batch Matrix Multiplacation
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))

out_bmm = torch.bmm(tensor1, tensor2)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2 ##x2 will be expended and to each row in x1, x2 will be added

# Other usefull operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
print(mean_x)

torch.sort(y, dim=0, descending=False)
print(y)
z = torch.clamp(x, min=0, max=10) #general case of special Relu

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x) #checks if at least one is true
z = torch.all(x) #checks if all are true










