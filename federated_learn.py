# from model import *
# from data import *
# from main import client_epochs, client_batch_size, fed_epochs, fed_clients
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import warnings
#
# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#
#
# def fed_average_weights(models):
#     weights_arrays = []
#     for model in models:
#         weights = model.get_weights()
#         weights_arrays.append(weights)
#     average_weights = np.average(weights_arrays, 0)
#     return average_weights
#
#
# def client_train(model, train_data):
#     user_id, item_id, rating = datatoinput(train_data)
#     history = model.fit([np.array(user_id), np.array(item_id)], np.array(rating), batch_size=client_batch_size,
#                         epochs=client_epochs, verbose=0, shuffle=True)
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
#
#
# def federated_learning(userlist, itemlist, traindata, validdata, testdata):
#     global weights, fedavg_model, avg_loss, train_avg_maes, train_avg_mses, test_avg_maes, test_avg_mses
#     n_users, n_items = len(userlist), len(itemlist)
#     base_modle = GMF(n_users, n_items)
#     # T is total fed training nums
#     for t in range(fed_epochs):
#         print("T" + str(t + 1) + " start, random seed=" + str(t + 1))
#         np.random.seed(t + 1)
#         loss_s, modle_s = [], []
#         mae_s, mse_s = [], []
#         # every loops random choose clients to join fed_train
#         cli = np.random.choice(n_users, fed_clients, replace=False)
#         for i in range(fed_clients):
#             model_i = base_modle
#             all_data = np.vstack((traindata, validdata))
#             train_data, valid_data = split_data_to_train_valid(all_data, cli[i])
#             # train_data = get_client_data(traindata, cli[i])
#             # valid_data = get_client_data(validdata, cli[i])
#             model_i, loss = client_train(model_i, train_data)
#             mae, mse = client_test(model_i, valid_data)
#             modle_s.append(model_i)
#             loss_s.append(loss)
#             mae_s.append(mae)
#             mse_s.append(mse)
#             if i % 100 == 0:
#                 print("t{}, loss: {:.5f} mae: {:.5f} mse: {:.5f}".format(t + 1, np.average(loss_s), np.average(mae_s), np.average(mse_s)))
#         avg_loss = np.average(loss_s)
#         train_avg_maes = np.average(mae_s)
#         train_avg_mses = np.average(mse_s)
#         log_msg = 'T{}, Train: avg_loss: {} avg_mae: {:.5f} avg_mse: {:.5f}'.format(t + 1, avg_loss, train_avg_maes,
#                                                                                       train_avg_mses)
#         write_log(log_msg)
#         print(log_msg)
#
#         # get fed_avg modle
#         avg_weights = fed_average_weights(modle_s)
#         base_modle.set_weights(avg_weights)
#         # Test
#         # test1: test all clients in test_data random
#         # test1: test clients which not join the fed trainning
#         # test3：randon choose clients from test_data to test
#
#         # test_cli_num = int(num_of_clients * 0.4)
#         mae, mse = client_test(base_modle, testdata)
#         mae_s.clear(), mae_s.append(mae)
#         mse_s.clear(), mse_s.append(mse)
#
#         test_avg_maes = (np.average(mae_s))
#         test_avg_mses = (np.average(mse_s))
#         log_msg = 'T{}, Test: avg_mae: {:.5f} avg_mse: {:.5f}'.format(t + 1, test_avg_maes, test_avg_mses)
#         write_log(log_msg)
#         # save fed_avg modle
#         modle_name = str("savedmodle/" + str(fed_clients) + "_" + str(t + 1) + '.h5')
#         base_modle.save(modle_name)
#         print(log_msg, '\n')
#
#     return avg_loss, train_avg_maes, train_avg_mses, test_avg_maes, test_avg_mses
#
#
