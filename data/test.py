from model import *
import collections

model_1 = GMF(2, 3)
user = torch.tensor(0)
item = torch.tensor(2)
rating = torch.tensor(5)
# if torch.cuda.is_available():
#     print('cuda is avaliable')
#     model_1.cuda()
#
#     user = user.cuda()
#     item = user.cuda()
#     rating = user.cuda()

# 打印模型各层的名称
# outputs:
# MF_Embedding_User.weight
# MF_Embedding_Item.weight
# linear.weight
# linear.bias
for name in model_1.state_dict():
    print(name)

# 打印模型各层的名称和对应的参数
print("模型各层的名称和对应的参数: ")
print(model_1.state_dict(), '\n')
# OrderedDict([
# ('MF_Embedding_User.weight', tensor([[ 3.5796, -0.9960], [ 0.8931, -0.8351]], device='cuda:0')),
# ('MF_Embedding_Item.weight', tensor([[ 1.3880,  1.1688], [ 1.8975, -0.2879], [-1.4561,  1.0703]], device='cuda:0'))
# ('linear.weight', tensor([[ 0.5277, -0.4340]], device='cuda:0')), ('linear.bias', tensor([-0.0121], device='cuda:0'))])


# 打印第一层名称和对应参数
# name: MF_Embedding_User.weight
# data: tensor([[ 1.5993,  0.9157], [-0.6292, -0.3372]], device='cuda:0')
print("某一层名称和对应参数: ")
params = list(model_1.named_parameters())
print("name: ", params[0][0])  # name
print("data: ", params[0][1].data)  # data -->params[0][1].datadetach().cpu().numpy()
print('\n')

# 将layer名称和参数对应放入字典
# dict[name, data]， 可以根据name打印data
params = {}  # change the tpye of 'generator' into dict
for name, param in model_1.named_parameters():
    params[name] = param.detach().cpu().numpy()

print("GMF_User_Embedding.weight: \n", params['GMF_User_Embedding.weight'], "\n")

# print("model.modules()")
# # scheme4
# for layer in model.modules():
#     #print(layer)
#     if (isinstance(layer, nn.Module.MF_Embedding_User)):
#          print("-")

# # 打印每一层的参数名和参数值
# # schemem1(recommended)
print("打印每一层的参数名和参数值: \n")
print("方法1 : ")
for name, param in model_1.named_parameters():
    print(name, param)

# # scheme2
print("方法2 : ")
for name in model_1.state_dict():
    print(name)
    print(model_1.state_dict()[name])

num_layers = 3
factor_num = 32
for i in range(num_layers):
    print(factor_num * (2 ** (num_layers - i)))

MLP_modules = []
for i in range(num_layers):
    input_size = factor_num * (2 ** (num_layers - i))  # 64,32,16
    #MLP_modules.append(nn.Dropout(p=self.dropout))
    MLP_modules.append(nn.Linear(input_size, input_size // 2))
    MLP_modules.append(nn.ReLU())

MLP_layers = nn.Sequential(*MLP_modules)
print("ncf_MLP_layers")
print(MLP_layers)

mlp = MLP(1,2)
print("mlp: ")
print(mlp)

