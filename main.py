from data import readdata, write_log
from torch_federated_learn import federated_learn

# filename1 = 'data/ml.userlist'
# filename2 = 'data/ml.itemlist'
# filename3 = 'data/ml.train.rating'
# filename4 = 'data/ml.valid.rating'
# filename5 = 'data/ml.test.rating'

filename1 = 'data/ml0.userlist'
filename2 = 'data/ml0.itemlist'
filename3 = 'data/ml0.train.rating'
filename4 = 'data/ml0.valid.rating'
filename5 = 'data/ml0.test.rating'

userlist, itemlist, traindata, validdata, testdata = readdata(filename1, filename2, filename3, filename4, filename5)


def train(userlist, itemlist, traindata, validdata, testdata):
    # log_msg = str(_fed_clients_num) + " NUM OF CLIENTS " + " START"
    log_msg = " NUM OF CLIENTS " + " START"
    print(log_msg)
    write_log(log_msg)
    avg_loss, train_avg_maes, test_avg_maes, train_avg_mses, test_avg_mses = federated_learn(userlist, itemlist,traindata, validdata, testdata)

    log_msg = " NUM OF CLIENTS " + " END"
    # log_msg = str(_fed_clients_num) + " NUM OF CLIENTS " + " END"
    print(log_msg)
    print("avg_loss: ", avg_loss)
    print("avg_maes_train: ", train_avg_maes)
    print("train_avg_maes: ", test_avg_maes)
    print("avg_mses_train: ", train_avg_mses)
    print("test_avg_mses: ", test_avg_mses)
    write_log(log_msg)


if __name__ == '__main__':
    train(userlist, itemlist, traindata, validdata, testdata)
