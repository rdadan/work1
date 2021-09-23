from model import *
from data import *  # split_data_to_train_valid, write_log
from torch_local_train import *
import heapq
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# hyper-parameter
_learn_rate = 0.001
_fed_epochs = 2
_client_epochs = 1
_fed_clients_num = 2  # int(0.50 * len(userlist))  # %
_client_batch_size = 2


def federated_learn(userlist, itemlist, traindata, validdata, testdata):
    print(" -------------- Fed Start, Total " + str(_fed_clients_num) + " Cients Join Fed " + " -------------- ")
    global ncf_avg, avg_loss
    train_avg_maes = []
    train_avg_mses = []

    n_users, n_items = len(userlist), len(itemlist)
    use_cuda = is_use_cuda()

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

        # every loops random choose clients to join fed_train
        cli = np.random.choice(n_users, _fed_clients_num, replace=False)
        """ num of clis join fed train """
        for i in range(_fed_clients_num):
            gmf_i = gmf
            mlp_i = mlp
            all_data = np.vstack((traindata, validdata))
            cli_train_data, cli_valid_data = split_data_to_train_valid(all_data, cli[i])

            """ cli_i local train gmf """
            gmf_i, gmf_loss_i = cli_local_train(gmf_i, optimizer_gmf, cli_train_data, _client_epochs,
                                                _client_batch_size, use_cuda)
            """ cli_i local train mlp """
            mlp_i, mlp_loss_i = cli_local_train(mlp_i, optimizer_mlp, cli_train_data, _client_epochs,
                                                _client_batch_size, use_cuda)

            """ local fed,  cli_i local train ncf  """
            """ updata ncf_i with gmf_i, mlp_i"""
            ncf_i = NCF(n_users, n_items, gmf_i, mlp_i)
            if use_cuda:
                ncf_i.cuda()

            optimizer = optim.SGD(ncf_i.parameters(), lr=0.001)
            ncf_i, ncf_loss_i = cli_local_train(ncf_i, optimizer, cli_train_data, _client_epochs,
                                                _client_batch_size, use_cuda)

            ncf_s.append(ncf_i)
            ncf_loss_s.append(ncf_loss_i)
            if (i + 1) % 10 == 0:
                print("Fed T_{}, first {} clents' avg loss: {:.5f} ".format(t + 1, i + 1, np.average(ncf_loss_s)))

        """ fed t over, get fed avg_ncf  """
        avg_loss = np.average(ncf_loss_s)
        ncf_avg = get_average_weights(ncf_s)

        """
        Test ncf_avg modle with validdata, 
        [test all clients in test_data random]
        [clients which not join the fed trainning]
        [randon choose clients from test_data to test]
        """
        print(" -------------- Fed T_{} Over, Test Modle with Validdata  -------------- ".format(t+1))
        mae, mse = cli_local_test(ncf_avg, validdata, _client_batch_size, use_cuda)
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

        """ one loop fed over, updata client gmf/mlp user/item embedding with fed_ncf_avg user/item embedding"""
        gmf = update_local_embedding(gmf, ncf_avg, name='gmf')
        mlp = update_local_embedding(mlp, ncf_avg, name='mlp')

    """ fed learning over, test final fed_ncf_avg modle with testdata """
    print(" -------------- Fed Over, Test Fed_Avg_Modle  -------------- ")
    test_avg_maes, test_avg_mses = cli_local_test(ncf_avg, testdata, _client_batch_size, use_cuda)
    print("avg_loss: ", avg_loss)
    print("min train_avg_maes: ", heapq.nsmallest(1, train_avg_maes))
    print("test_avg_maes: ", test_avg_maes)
    print("min train_avg_mses: ", heapq.nsmallest(1, train_avg_mses))
    print("test_avg_mses: ", test_avg_mses)
    # return avg_loss, train_avg_maes, test_avg_maes, train_avg_mses, test_avg_mses
