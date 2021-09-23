import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
import random
import torch




def load_data():
    ratings_df = pd.read_csv('ratings.dat', sep='::', header=None, engine='python')
    # ratings_df = pd.read_csv('./ratings.csv', sep=',', header=None, engine='python')
    ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    # print("ratings_fd:  \n", ratings_df[:5])
    # print(len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique()))

    train, test = train_test_split(ratings_df, test_size=0.2, random_state=7856)
    # n_users, n_movies = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    # print(n_users, n_movies)
    # x_train = x_train.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

    n_users, n_movies = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    return train.values, test.values, n_users, n_movies

def readdata(filename1, filename2, filename3, filename4, filename5):
    userlist = []
    with open(filename1, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.strip()
        userlist.append(int(line))
    itemlist = []
    with open(filename2, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.strip()
        itemlist.append(int(line))
    traindata = []
    with open(filename3, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.split('\t')
        user = int(line[0])
        item = int(line[1])
        rating = float(line[2])
        traindata.append((user, item, rating))
    validdata = []
    with open(filename4, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.split('\t')
        user = int(line[0])
        item = int(line[1])
        rating = float(line[2])
        validdata.append((user, item, rating))
    testdata = []
    with open(filename5, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.split('\t')
        user = int(line[0])
        item = int(line[1])
        rating = float(line[2])
        testdata.append((user, item, rating))
    return userlist, itemlist, traindata, validdata, testdata


def split_data(data, cli_idx, size):
    tmp = np.array(data).astype(np.int32)
    tmp = tmp[tmp[:, 0] == cli_idx]
    np.random.shuffle(tmp)
    size = int(tmp.shape[0] * size)
    data_1, data_2 = tmp[:size], tmp[size:-1]
    return data_1, data_2


def get_client_data(data, cli_idx):
    tmp = np.array(data).astype(np.int32)
    tmp = tmp[tmp[:, 0] == cli_idx]
    # np.random.shuffle(tmp)
    # tf.convert_to_tensor(tmp)
    return tmp


def datatoinput(data, use_cuda):
    users = []
    items = []
    ratings = []
    for example in data:
        users.append(example[0])
        items.append(example[1])
        ratings.append(example[2])
    users = torch.tensor(users, dtype=torch.int64)
    items = torch.tensor(items, dtype=torch.int64)
    ratings = torch.tensor(ratings, dtype=torch.float32)
    if use_cuda:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
    return users, items, ratings


def batchtoinput(batch, use_cuda):
    users = []
    items = []
    ratings = []
    for example in batch:
        users.append(example[0])
        items.append(example[1])
        ratings.append(example[2])
    users = torch.tensor(users, dtype=torch.int64)
    items = torch.tensor(items, dtype=torch.int64)
    ratings = torch.tensor(ratings, dtype=torch.float64)
    if use_cuda:
        users = users.cuda()
        items = items.cuda()
        ratings = ratings.cuda()
    return users, items, ratings


def getbatches(traindata, batch_size, use_cuda, shuffle):
    dataset = traindata.copy()
    if shuffle:
       random.shuffle(dataset)
    for batch_i in range(0, int(np.ceil(len(dataset) / batch_size))):
        start_i = batch_i * batch_size
        batch = dataset[start_i:start_i + batch_size]
        yield batchtoinput(batch, use_cuda)

def time_detal():
    # 年/月/日
    daytime = datetime.datetime.now().strftime('%Y/%m/%d')
    # 时：分：秒
    hourtime = datetime.datetime.now().strftime("%H:%M:%S")
    detail_time = daytime + " " + hourtime

    return detail_time


def write_log(log_msg):
    fp = open("mae_mas_log.txt", mode='a+')
    time = time_detal()
    time = '--------------------------------- ' + time + ' ---------------------------------'
    fp.write(time + '\n')
    fp.write(log_msg + '\n')
    fp.close()
