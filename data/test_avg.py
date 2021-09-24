from model import *
import collections

# model_1 = GMF(1, 2)
# model_2 = GMF(1, 2)
# models = [model_1, model_2]
# avg_modle = models[0]
# print("平均两个相同模型的参数 ")
# print("model_1 : ")
# for name in model_1.state_dict():
#     print(name)
#     print(model_1.state_dict()[name])
#
# print("model_2 : ")
# for name in model_2.state_dict():
#     print(name)
#     print(model_2.state_dict()[name])
#
# print("test avg modle")
# worker_state_dict = [x.state_dict() for x in models]
# weight_keys = list(worker_state_dict[0].keys())
# fed_state_dict = collections.OrderedDict()
# for key in weight_keys:
#     key_sum = 0
#     for i in range(len(models)):
#         key_sum = key_sum + worker_state_dict[i][key]
#     fed_state_dict[key] = key_sum / len(models)
#
# # update fed weights to avg_modle
# avg_modle.load_state_dict(fed_state_dict)
#
# print("avg_modle : ")
# for name in avg_modle.state_dict():
#     print(name)
#     print(np.array(avg_modle.state_dict()[name]))

print("----------")
gmf = GMF(1, 2)
# mlp = MLP_1(1, 2)
print(gmf)
#print(mlp)
#
# ncf = NCF(1, 2, gmf_1=gmf, gmf_2=gmf)
# print(ncf)


# filename1 = 'ml0.userlist'
# filename2 = 'ml0.itemlist'
# filename3 = 'ml0.train.rating'
# filename4 = 'ml0.valid.rating'
# filename5 = 'ml0.test.rating'
filename1 = 'ml0.userlist'
filename2 = 'ml0.itemlist'
filename3 = 'ml0.train.rating'
filename4 = 'ml0.valid.rating'
filename5 = 'ml0.test.rating'

from data import readdata, getbatches
userlist, itemlist, traindata, validdata, testdata = readdata(filename1, filename2, filename3, filename4, filename5)
print("111-------------------------\n")

mlp = MLP_1(len((userlist)), len(itemlist))
gmf = GMF(len((userlist)), len(itemlist))
net = GMF_FED(len(userlist), len(itemlist), gmf_1=gmf, gmf_2=gmf)
print(net)


net.apply(weights_init)
use_cuda = is_use_cuda()
if use_cuda:
    net.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
net.train()
error = 0
num = 0


for k, (users, items, ratings) in enumerate(getbatches(traindata, 256, use_cuda, True)):
    # set gradient 0
    optimizer.zero_grad()
    # forward propagation for loss
    pred = net(users, items)
    loss_ = net.get_loss(pred, ratings)
    # back propagation for gradient
    loss_.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
    optimizer.step()

    # get avg err
    error += loss_.detach().cpu().numpy() * len(users)
    num += len(users)

print("2222-----------------------")

