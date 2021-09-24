import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import collections
import os

os.environ['CUDA_ENABLE_DEVICES'] = '0'

Layers = [16, 128, 64, 32]


class GMF(nn.Module):
    def __init__(self, num_users, num_items):
        super(GMF, self).__init__()
        self.GMF_User_Embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=Layers[0])
        self.GMF_Item_Embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=Layers[0])

        """ linear logistic regression layer """
        self.gmf_predict_layer = nn.Linear(Layers[0], 1)

    def forward(self, user, item):
        """ embedding"""
        GMF_User_Embedding = self.GMF_User_Embedding(user)
        GMF_Item_Embedding = self.GMF_Item_Embedding(item)
        # dot product, multiply the elements at the corresponding positions
        embedding_vec = torch.mul(GMF_User_Embedding, GMF_Item_Embedding)

        """ linear logistic regression """
        pred = self.gmf_predict_layer(embedding_vec)
        outputs = F.relu(pred)
        return outputs

    def get_loss(self, pred, rating):
        return torch.mean(torch.pow(pred - rating, 2))  # + self._lambda * regularizer


class GMF_FED(nn.Module):
    def __init__(self, user_num, item_num, gmf_1, gmf_2):
        super(GMF_FED, self).__init__()
        # layers = [32, 256, 128, 64, 32]
        layers = Layers
        # Embedding
        self.GMF_1_User_Embedding = nn.Embedding(num_embeddings=user_num, embedding_dim=layers[0])
        self.GMF_1_Item_Embedding = nn.Embedding(num_embeddings=item_num, embedding_dim=layers[0])
        self.GMF_2_User_Embedding = nn.Embedding(num_embeddings=user_num, embedding_dim=layers[0])
        self.GMF_2_Item_Embedding = nn.Embedding(num_embeddings=item_num, embedding_dim=layers[0])
        # Pred
        self.fed_predict_layer = nn.Linear(layers[0] * 2, 1)

        # Init
        self._init_weight_(gmf_1, gmf_2)

    def forward(self, user_inputs, item_inputs):
        """ embedding"""
        GMF_1_User_Embedding = self.GMF_1_User_Embedding(user_inputs)
        GMF_1_Item_Embedding = self.GMF_1_Item_Embedding(item_inputs)
        GMF_2_User_Embedding = self.GMF_2_User_Embedding(user_inputs)
        GMF_2_Item_Embedding = self.GMF_2_Item_Embedding(item_inputs)

        # dot product, multiply the elements at the corresponding positions
        Fed_User_Embedding = torch.cat((GMF_1_User_Embedding, GMF_2_User_Embedding), dim=1)
        Fed_Item_Embedding = torch.cat((GMF_1_Item_Embedding, GMF_2_Item_Embedding), dim=1)
        embedding_vec = torch.mul(Fed_User_Embedding, Fed_Item_Embedding)

        """ linear logistic regression """
        pred = self.fed_predict_layer(embedding_vec)
        outputs = F.relu(pred)
        return outputs
        # return prediction.view(-1)

    def _init_weight_(self, gmf_1, gmf_2):
        """ weights initialization. """
        # init ncf embedding layers with gmf,mlp embedding
        self.GMF_1_User_Embedding.weight.data.copy_(gmf_1.GMF_User_Embedding.weight)
        self.GMF_1_Item_Embedding.weight.data.copy_(gmf_1.GMF_Item_Embedding.weight)
        self.GMF_2_User_Embedding.weight.data.copy_(gmf_2.GMF_User_Embedding.weight)
        self.GMF_2_Item_Embedding.weight.data.copy_(gmf_2.GMF_Item_Embedding.weight)


        predict_weight = torch.cat((gmf_1.gmf_predict_layer.weight, gmf_2.gmf_predict_layer.weight), dim=1)
        precit_bias = gmf_1.gmf_predict_layer.bias + gmf_2.gmf_predict_layer.bias
        self.fed_predict_layer.weight.data.copy_(predict_weight)
        self.fed_predict_layer.bias.data.copy_(precit_bias)

    def get_loss(self, pred, rating):
        return torch.mean(torch.pow(pred - rating, 2))


class MLP_1(nn.Module):
    def __init__(self, num_users, num_items):
        super(MLP_1, self).__init__()
        layers = Layers
        # layers = [2, 2, 2, 2, 2]
        self.MLP_User_Embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
        self.MLP_Item_Embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
        self.dropout = 0
        """ fully connected layer """
        self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        """ embed cat"""
        self.mlp_predict_layer = nn.Linear(layers[-1], 1)
        # self.activation = nn.Sigmoid()

    def forward(self, user, item):
        """ Embedding """
        MLP_User_Embedding = self.MLP_User_Embedding(user)
        MLP_Item_Embedding = self.MLP_Item_Embedding(item)

        """ horizontal concatenation """
        embedding_vec = torch.cat((MLP_User_Embedding, MLP_Item_Embedding), dim=-1)
        # print("embedding_vec：", embedding_vec.size())
        """ fully connected """
        for linear in self.MLP_Layers:
            embedding_vec = linear(embedding_vec)
            embedding_vec = F.relu(embedding_vec)
        # logistic regression
        pred = self.mlp_predict_layer(embedding_vec)
        # output = self.activation(embedding_vec)
        outputs = F.relu(pred)
        return outputs

    def get_loss(self, pred, rating):
        return torch.mean(torch.pow(pred - rating, 2))


class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers_num=3, factor_num=2):
        super(MLP, self).__init__()

        self.MLP_User_Embedding = nn.Embedding(num_embeddings=num_users,
                                               embedding_dim=factor_num // 2)  # factor_num * (2 ** (layers_num - 1)))
        self.MLP_Item_Embedding = nn.Embedding(num_embeddings=num_items,
                                               embedding_dim=factor_num // 2)  # factor_num * (2 ** (layers_num - 1)))
        self.dropout = 0
        """ fully connected layer """
        MLP_modules = []  # [256,128],[128,64],[64,32]
        for i in range(layers_num):
            input_size = factor_num * (2 ** (layers_num - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_Layers = nn.Sequential(*MLP_modules)

        # -2-
        # layers = [32, 256, 128, 64, 32]
        # self.MLP_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])

        self.mlp_predict_layer = nn.Linear(factor_num, 1)  # [32,1]
        self.activation = nn.Sigmoid()

    def forward(self, user, item):
        """ Embedding """
        MLP_User_Embedding = self.MLP_User_Embedding(user)
        MLP_Item_Embedding = self.MLP_Item_Embedding(item)

        """ horizontal concatenation """
        embedding_vec = torch.cat([MLP_User_Embedding, MLP_Item_Embedding], dim=-1)

        """ fully connected """
        for linear in self.MLP_Layers:
            embedding_vec = linear(embedding_vec)
            embedding_vec = F.relu(embedding_vec)
        # logistic regression
        embedding_cat = self.mlp_predict_layer(embedding_vec)
        output = self.activation(embedding_cat)
        return output

    def get_loss(self, pred, rating):
        return torch.mean(torch.pow(pred - rating, 2))


class NCF(nn.Module):
    def __init__(self, user_num, item_num, gmf_1=None, gmf_2=None,
                 flag='Local_Fed', factor_num=Layers[0], num_layers=3, dropout=0, ):
        super(NCF, self).__init__()
        # layers = [32, 256, 128, 64, 32]
        layers = Layers
        self.dropout = dropout
        self.flag = flag
        self.gmf_1 = gmf_1
        self.gmf_2 = gmf_2
        self.activation = nn.Sigmoid()
        # Embedding
        # self.GMF_User_Embedding = nn.Embedding(num_embeddings=user_num, embedding_dim=layers[0])
        # self.GMF_Item_Embedding = nn.Embedding(item_num, layers[0])
        # self.MLP_User_Embedding = nn.Embedding(user_num, layers[0]//2)#factor_num * (2 ** (num_layers - 1)))  # 32 *(2**2)=128
        # self.MLP_Item_Embedding = nn.Embedding(item_num, layers[0]//2)#factor_num * (2 ** (num_layers - 1)))
        self.NCF_User_Embedding = nn.Embedding(num_embeddings=user_num, embedding_dim=layers[0] * 2)
        self.NCF_Item_Embedding = nn.Embedding(num_embeddings=user_num, embedding_dim=layers[0] * 2)
        # self.NCF_FC_Layers = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        # ncf full connect layers
        # NCF_Full_Connect = []
        # for i in range(num_layers):
        #     input_size = factor_num * (2 ** (num_layers - i))
        #     NCF_Full_Connect.append(nn.Dropout(p=self.dropout))
        #     NCF_Full_Connect.append(nn.Linear(input_size, input_size // 2))
        #     NCF_Full_Connect.append(nn.ReLU())
        # self.NCF_FC_Layers = nn.Sequential(*NCF_Full_Connect)
        # FC Layers
        self.NCF_FC_Layers = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
        # self.linear = nn.Linear(layers[-1], layers[0])

        """ embed mul"""
        # self.ncf_predict_layer = nn.Linear(layers[0]*2, 1)  # [32,1]
        self.ncf_predict_layer = nn.Linear(layers[-1], 1)
        self._init_weight_()

    def forward(self, user, item):
        global GMF_Embedding, MLP_Embedding, NCF_Embedding
        # if not self.flag == 'GMF':
        # GMF_User_Embedding = self.GMF_User_Embedding(user)
        # GMF_Item_Embedding = self.GMF_Item_Embedding(item)
        # #GMF_Embedding = torch.mul(GMF_User_Embedding, GMF_Item_Embedding)
        # #if not self.flag == 'MLP':
        # MLP_User_Embedding = self.MLP_User_Embedding(user)
        # MLP_Item_Embedding = self.MLP_Item_Embedding(item)
        # #MLP_Embedding = torch.cat((MLP_User_Embedding, MLP_Item_Embedding), -1)
        # #output_MLP = self.NCF_FC_Layers(embedding_vec)
        # NCF_User_Embedding = torch.cat((GMF_User_Embedding, MLP_User_Embedding), dim=-1) # 8+16=24
        # NCF_Item_Embedding = torch.cat((GMF_Item_Embedding, MLP_Item_Embedding), dim=-1)
        NCF_Embedding = torch.mul((self.NCF_User_Embedding, self.NCF_Item_Embedding))
        """ fully connected """
        for linear in self.NCF_FC_Layers:
            NCF_Embedding = linear(NCF_Embedding)
            NCF_Embedding = F.relu(NCF_Embedding)

        # if self.flag == 'GMF':
        #     concat = GMF_Embedding
        # elif self.flag == 'MLP':
        #     concat = MLP_Embedding
        # elif self.flag == 'Local_Fed':
        #     concat = torch.cat((GMF_Embedding, MLP_Embedding), -1)

        pred = self.ncf_predict_layer(NCF_Embedding)
        # outputs = self.activation(prediction)
        outputs = F.relu(pred)
        return outputs
        # return prediction.view(-1)

    def _init_weight_(self):
        """ weights initialization. """
        # if not self.flag == 'Local_Fed':
        #     nn.init.normal_(self.GMF_User_Embedding.weight, std=0.01)
        #     nn.init.normal_(self.MLP_User_Embedding.weight, std=0.01)
        #     nn.init.normal_(self.GMF_Item_Embedding.weight, std=0.01)
        #     nn.init.normal_(self.MLP_Item_Embedding.weight, std=0.01)
        #
        #     for m in self.NCF_FC_Layers:
        #         if isinstance(m, nn.Linear):
        #             nn.init.xavier_uniform_(m.weight)
        #     nn.init.kaiming_uniform_(self.ncf_predict_layer.weight, a=1, nonlinearity='sigmoid')
        #
        #     for m in self.modules():
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             m.bias.data.zero_()
        # else:
        # init ncf embedding layers with gmf,mlp embedding
        NCF_User_Embedding_Weight = torch.cat(
            (self.gmf_1.GMF_User_Embedding.weight + self.gmf_2.GMF_User_Embedding.weight), dim=-1)
        NCF_Item_Embedding_Weight = torch.cat(
            (self.gmf_1.GMF_Item_Embedding.weight + self.gmf_2.GMF_Item_Embedding.weight), dim=-1)
        self.NCF_User_Embedding.weight.data.copy_(NCF_User_Embedding_Weight)
        self.NCF_Item_Embedding.weight.data.copy_(NCF_Item_Embedding_Weight)
        # self.MLP_User_Embedding.weight.data.copy_(
        #     self.gmf_2.MLP_User_Embedding.weight)
        # self.MLP_Item_Embedding.weight.data.copy_(
        #     self.gmf_2.MLP_Item_Embedding.weight)

        # print("-----init ncf--------------: ")
        # print(self.GMF_User_Embedding.weight.data)
        # print(self.GMF_Item_Embedding.weight.data)
        # print(self.MLP_User_Embedding.weight.data)
        # print(self.MLP_Item_Embedding.weight.data)
        # print("-------init ncf over-----------: \n ")
        # init ncf fully connect layers
        # for (m1, m2) in zip(self.NCF_FC_Layers, self.gmf_2.MLP_Layers):
        #     if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
        #         m1.weight.data.copy_(m2.weight)
        #         m1.bias.data.copy_(m2.bias)

        # init ncf predict layers
        predict_weight = torch.cat((self.gmf_1.gmf_predict_layer.weight, self.gmf_2.mlp_predict_layer.weight), dim=1)
        precit_bias = self.gmf_1.gmf_predict_layer.bias + self.gmf_2.mlp_predict_layer.bias
        self.ncf_predict_layer.weight.data.copy_(predict_weight)
        self.ncf_predict_layer.bias.data.copy_(precit_bias)
        # print("predict_weight: " , predict_weight.size())

    # print("precit_bias: " , precit_bias)
    # print("ncf_predict_layer.weight: " , self.ncf_predict_layer.weight.data.size())

    def get_loss(self, pred, rating):
        return torch.mean(torch.pow(pred - rating, 2))


# class NeuralMF(nn.Module):
#
#     def __init__(self, num_users, num_items, outputs_dim=10, layers=[20, 64, 32, 16, 8]):
#         super(NeuralMF, self).__init__()
#
#         """ embedding layer of GMF """
#         self.GMF_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=outputs_dim)
#         self.GMF_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=outputs_dim)
#
#         """ embedding layer of MLP """
#         self.MLP_Embedding_User = nn.Embedding(num_embeddings=num_users, embedding_dim=layers[0] // 2)
#         self.MLP_Embedding_Item = nn.Embedding(num_embeddings=num_items, embedding_dim=layers[0] // 2)
#
#         """ fully connect layer """
#         self.fully_connect_layer = nn.ModuleList(
#             [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])
#         self.linear = nn.Linear(layers[-1], outputs_dim)
#
#         self.linear2 = nn.Linear(2 * outputs_dim, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, user, item):
#         """ GMF embedding """
#         GMF_Embedding_User = self.GMF_Embedding_User(user)
#         GMF_Embedding_Item = self.GMF_Embedding_Item(item)
#         """ MLP embedding """
#         MLP_Embedding_User = self.MLP_Embedding_User(user)
#         MLP_Embedding_Item = self.MLP_Embedding_Item(item)
#
#         #
#         gmf_vec = torch.mul(GMF_Embedding_User, GMF_Embedding_Item)
#
#         mlp_cat = torch.cat([MLP_Embedding_User, MLP_Embedding_Item], dim=-1)
#         for linear in self.fully_connect_layer:
#             mlp_cat = linear(mlp_cat)
#             mlp_cat = F.relu(mlp_cat)
#         mlp_vec = self.linear(mlp_cat)
#
#         neuMF_vector = torch.cat([gmf_vec, mlp_vec], dim=-1)
#         # liner
#         linear = self.linear2(neuMF_vector)
#         output = self.sigmoid(linear)
#
#         return output
#
#     def set_local_embedding(self, gmf, mlp):
#         """A simple implementation of load pretrained parameters """
#         self.GMF_Embedding_User.weight.data = gmf.GMF_Embedding_User.weight.data
#         self.GMF_Embedding_Item.weight.data = gmf.GMF_Embedding_Item.weight.data
#         self.MLP_Embedding_User.weight.data = mlp.MLP_Embedding_User.weight.data
#         self.MLP_Embedding_Item.weight.data = mlp.MLP_Embedding_Item.weight.data
#
# for (m1, m2) in zip(self.mlp_layers.mlp_layers, mlp.mlp_layers.mlp_layers):
#     if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
#         m1.weight.data.copy_(m2.weight)
#         m1.bias.data.copy_(m2.bias)
#
# predict_weight = torch.cat([gmf.ncf_predict_layer.weight, mlp.ncf_predict_layer.weight], dim=1)
# predict_bias = gmf.ncf_predict_layer.bias + mlp.ncf_predict_layer.bias
#
# self.ncf_predict_layer.weight.data.copy_(0.5 * predict_weight)
# self.ncf_predict_layer.weight.data.copy_(0.5 * predict_bias)

# def Base_Model(num_users, num_items):
#     n_latent_factors_user = 32
#     n_latent_factors_item = 32
#
#     user_input = keras.layers.Input(shape=[1], name='User')
#     user_embedding = keras.layers.Embedding(num_users, n_latent_factors_user, name='UserEmbedding')(user_input)
#     user_vec = keras.layers.Flatten(name='FlattenUsers')(user_embedding)
#     #user_vec = keras.layers.Dropout(0.2)(user_vec)
#
#     item_input = keras.layers.Input(shape=[1], name='Item')
#     item_embedding = keras.layers.Embedding(num_items, n_latent_factors_item, name='MovieEmbedding')(item_input)
#     movie_vec = keras.layers.Flatten(name='FlattenMovies')(item_embedding)
#     #movie_vec = keras.layers.Dropout(0.2)(movie_vec)
#
#     concat = keras.layers.concatenate([movie_vec, user_vec])
#     concat_dropout = keras.layers.Dropout(0.2)(concat)
#
#     dense = keras.layers.Dense(200, name='FullyConnected')(concat)
#     dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
#
#     dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(concat)
#     dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
#
#     dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dense_2)
#     dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
#
#     dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)
#     result = keras.layers.Dense(1, activation='relu', name='Activation')(dense_4)
#
#     adam = Adam(lr=0.005)
#     model = keras.Model([user_input, item_input], result)
#     model.compile(optimizer=adam, loss='mean_absolute_error')
#     return model


def is_use_cuda():
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
    return use_cuda


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def tf_fed_average_weights(models):
    weights_arrays = []
    for model in models:
        weights = model.get_weights()
        weights_arrays.append(weights)
    average_weights = np.average(weights_arrays, 0)
    return average_weights


def get_average_weights(models):
    avg_modle = models[0]
    state_dict = [x.state_dict() for x in models]
    weight_keys = list(state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(models)):
            key_sum = key_sum + state_dict[i][key]
        fed_state_dict[key] = key_sum / len(models)
    # update fed weights to avg_modle
    avg_modle.load_state_dict(fed_state_dict)
    return avg_modle


def update_local_embedding(local_modle, fed_ncf, name):
    """ update gmf/mlp embeddings with trained fed embedding """
    if name == 'gmf':
        local_modle.GMF_User_Embedding.weight.data = fed_ncf.GMF_1_User_Embedding.weight.data
        local_modle.GMF_Item_Embedding.weight.data = fed_ncf.GMF_1_Item_Embedding.weight.data
    elif name == 'mlp':
        local_modle.MLP_User_Embedding.weight.data = fed_ncf.GMF_2_User_Embedding.weight.data
        local_modle.MLP_Item_Embedding.weight.data = fed_ncf.GMF_2_Item_Embedding.weight.data

    return local_modle
