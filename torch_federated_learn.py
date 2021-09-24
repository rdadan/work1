from model import *
from data import *  # split_data_to_train_valid, write_log
from torch_local_train import *
import heapq
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# hyper-parameter
_learn_rate = 0.001
_fed_epochs = 10
_client_epochs = 20
_client_batch_size = 128


def federated_learn(userlist, itemlist, traindata, validdata, testdata):

    _fed_clients_num = int(0.30 * len(userlist))
    use_cuda = is_use_cuda()
    print("use cuda：", use_cuda)
    print(" -------------- Fed Start, Total " + str(_fed_clients_num) + " Cients Join Fed " + " -------------- ")
    global ncf_avg, avg_loss
    train_avg_maes = []
    train_avg_mses = []

    n_users, n_items = len(userlist), len(itemlist)

    gmf = GMF(n_users, n_items)
    mlp = MLP_1(n_users, n_items)
    gmf.apply(weights_init)
    mlp.apply(weights_init)
    if use_cuda:
        gmf.cuda()
        mlp.cuda()

    optimizer_gmf = torch.optim.Adam(gmf.parameters(), lr=0.001, weight_decay=0.001)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=0.001)

    """  T is total fed epoches """
    for t in range(_fed_epochs):
        random.seed(t + 1)
        np.random.seed(t + 1)
        torch.manual_seed(t + 1)  # set random seed for cpu
        torch.cuda.manual_seed(t + 1)  # set random seed for current gpu
        torch.cuda.manual_seed_all(t + 1)  # set random seed for all gpus
        print(" -------------- Fed T_" + str(t + 1) + " start, random seed=" + str(t + 1) + " -------------- ")
        ncf_loss_s, ncf_s = [], []
        mae_s, mse_s = [], []
        all_public_testdata = np.empty(shape=(0,3))
        print(all_public_testdata)
        # every loops random choose clients to join fed_train
        cli = np.random.choice(n_users, _fed_clients_num, replace=False)
        """ num of clis join fed train """
        for i in range(_fed_clients_num):
            gmf_i = gmf
            mlp_i = gmf
            all_data = np.vstack((traindata, validdata))
            data_1, data_2 = split_data(all_data, cli[i], 0.5)
            data_1_private, data_1_public = split_data(data_1, cli[i], 0.7)
            """ cli_i local train gmf """
            gmf_i, gmf_loss_i = cli_local_train(gmf_i, optimizer_gmf, data_1_private, _client_epochs,
                                                _client_batch_size, use_cuda)
            """ cli_i local train mlp """
            data_2_private, data_2_public = split_data(data_2, cli[i], 0.7)
            mlp_i, mlp_loss_i = cli_local_train(mlp_i, optimizer_mlp, data_2_private, _client_epochs,
                                                _client_batch_size, use_cuda)

            """ local fed,  cli_i local train ncf  """
            """ updata ncf_i with gmf_i, mlp_i"""
            ncf_i = GMF_FED(n_users, n_items, gmf_i, mlp_i)
            if use_cuda:
                ncf_i.cuda()

            optimizer = optim.SGD(ncf_i.parameters(), lr=0.001)

            public_data = np.vstack((data_1_private, data_2_private)) # 非铭感数据
            public_train, public_test = split_data(public_data, cli[i], 0.8)

            ncf_i, ncf_loss_i = cli_local_train(ncf_i, optimizer, public_train, _client_epochs,
                                                _client_batch_size, use_cuda)

            ncf_s.append(ncf_i)
            ncf_loss_s.append(ncf_loss_i)
            test_mae, test_mse = cli_local_test(ncf_i, public_test, _client_batch_size, use_cuda)
            if (i + 1) % 10 == 0:
                 print("Fed T_{}, first {} clents' avg_loss: {:.5f}, test_mae: {:.5f}, test_mse: {:.5f} ".
                       format(t + 1, i + 1, np.average(ncf_loss_s), test_mae, test_mse))

            all_public_testdata = np.vstack((all_public_testdata, public_test))
        """ fed t over, get fed avg_ncf  """
        avg_loss = np.average(ncf_loss_s)
        ncf_avg = get_average_weights(ncf_s)

        """
        Test ncf_avg modle with validdata, 
        [test all clients in test_data random]
        [clients which not join the fed trainning]
        [randon choose clients from test_data to test]
        """
        print(" -------------- Fed T_{} Finish, Test Modle with All_Public_Testdata  -------------- ".format(t+1))
        mae, mse = cli_local_test(ncf_avg, all_public_testdata, _client_batch_size, use_cuda)
        mae_s.append(mae)
        mse_s.append(mse)
        avg_mae = np.average(mae_s)
        avg_mse = np.average(mse_s)
        train_avg_maes.append(avg_mae)
        train_avg_mses.append(avg_mse)
        log_msg = 'avg_mae: {:.5f}, avg_mse: {:.5f}'.format(t + 1, avg_mae, avg_mse)
        write_log(log_msg)
        # save fed_avg modle
        modle_name = str("savedmodle/" + str(_fed_clients_num) + "_" + str(t + 1) + '.pkl')
        torch.save(gmf, modle_name)
        print(log_msg, '\n')

        """ one loop fed finish, updata client gmf/mlp user/item embedding with fed_ncf_avg user/item embedding"""
        gmf = update_local_embedding(gmf, ncf_avg, name='gmf')
        mlp = update_local_embedding(mlp, ncf_avg, name='mlp')

    """ fed learning finish, test final fed_ncf_avg modle with testdata """
    print(" -------------- Fed Finish, Test Fed_Avg_Modle  -------------- ")
    test_avg_maes, test_avg_mses = cli_local_test(ncf_avg, testdata, _client_batch_size, use_cuda)
    print("avg_loss: ", avg_loss)
    print("min train_avg_maes: ", heapq.nsmallest(1, train_avg_maes))
    print("test_avg_maes: ", test_avg_maes)
    print("min train_avg_mses: ", heapq.nsmallest(1, train_avg_mses))
    print("test_avg_mses: ", test_avg_mses)
    # return avg_loss, train_avg_maes, test_avg_maes, train_avg_mses, test_avg_mses
