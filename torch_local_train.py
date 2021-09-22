import torch
import numpy as np
from data import getbatches, datatoinput
from evaluation import get_mae_mse
from sklearn.metrics import mean_squared_error, mean_absolute_error


def cli_local_train(net, optimizer, traindata, batch_size, epochs, use_cuda):

    loss_i = []
    for epoch in range(epochs):
        # switch to train mode
        net.train()
        error = 0
        num = 0
        for k, (users, items, ratings) in enumerate(getbatches(traindata, batch_size, use_cuda, True)):
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

        loss_i.append(error / num)

    return net, loss_i


def cli_local_test(net, testdata, batch_size, use_cuda):
    net.eval()  # switch to test mode
    ratlist = []
    predlist = []
    for k, (users, items, ratings) in enumerate(getbatches(testdata, batch_size, use_cuda, False)):
        pred = net(users, items)
        predlist.extend(pred.tolist())
        ratlist.extend(ratings.tolist())
    mae, mse = get_mae_mse(np.array(ratlist), np.array(predlist))

    return mae, mse


def cli_local_test_2(net, testdata, batch_size, use_cuda):
    # calculate auc pred model trained with each client
    user_id, item_id, true_rating_list = datatoinput(testdata, use_cuda)
    predict_rating_list = net.predict([np.array(user_id), np.array(item_id)], batch_size=64, verbose=0)

    mae = mean_absolute_error(true_rating_list, predict_rating_list)
    mse = mean_squared_error(true_rating_list, predict_rating_list)
    return mae, mse


def get_cli_embedding(model, flag):
    for name, weights in model.named_parameters():
        weights[name] = weights.detach().cpu().numpy()
        if flag == 0:
            return weights['GMF_User_Embedding.weight'], weights['GMF_Item_Embedding.weight']
        elif flag == 1:
            return weights['MLP_User_Embedding.weight'], weights['MLP_Item_Embedding.weight']

# def client_train(model, train_data, epochs, batch_size):
#     user_id, item_id, rating = datatoinput(train_data)
#     history = model.fit([np.array(user_id), np.array(item_id)], np.array(rating), batch_size=batch_size,
#                         epochs=epochs, verbose=0, shuffle=True)
#     loss = history.history["loss"][-1]
#     return model, loss
#
#
# def client_test(model, test_data):
#     # calculate auc pred model trained with each client
#     user_id, item_id, true_rating_list = datatoinput(test_data)
#     predict_rating_list = model.predict([np.array(user_id), np.array(item_id)], batch_size=64, verbose=0)
#     # mae, mse = get_eval(np.array(true_rating_list), np.array(predict_rating_list))
#     mae = mean_absolute_error(true_rating_list, predict_rating_list)
#     mse = mean_squared_error(true_rating_list, predict_rating_list)
#     return mae, mse
