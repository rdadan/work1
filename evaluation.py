from data import *
import heapq


# Hit Rating
def get_hit_ratio(ranklist, gt_item):
    for item in ranklist:
        if item == gt_item:
            return 1
        return 0


#  NDCG
def get_ndcg(ranklist, gt_item):
    for i in range(len(ranklist)):
        if ranklist[i] == gt_item:
            return np.log(2) / np.log(i + 2)
    return 0


# 一次评分预测
def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')

    test_data = torch.tensor(np.vstack([users, np.array(items)]).T)  # .to(device)
    pred = _model(test_data)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = pred[i]
    items.pop()

    # evaluate top rank list, heap sort, get Top K
    sort_key = lambda k: map_item_score[k]
    ranklist = heapq.nlargest(_K, map_item_score, key=sort_key)
    hr = get_hit_ratio(ranklist, gtItem)
    ndcg = get_ndcg(ranklist, gtItem)
    return hr, ndcg


def evaluate_model(model, testRatings, testNegatives, K):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    # if (num_thread > 1):  # Multi-thread
    #     pool = multiprocessing.Pool(processes=num_thread)
    #     res = pool.map(eval_one_rating, range(len(_testRatings)))
    #     pool.close()
    #     pool.join()
    #     hits = [r[0] for r in res]
    #     ndcgs = [r[1] for r in res]
    #     return (hits, ndcgs)

    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def get_mae_mse(ratlist, predlist, output=False):
    mae = np.mean(np.abs(ratlist - predlist))
    mse = np.mean(np.square(ratlist - predlist))
    # if output:
    #     maelist = np.abs(ratlist - predlist)
    #     with open('maelist.dat', 'w') as f:
    #         i = 0
    #         while i < len(maelist):
    #             f.write(str(maelist[i]) + '\n')
    #             i += 1
    #     mselist = np.square(ratlist - predlist)
    #     with open('mselist.dat', 'w') as f:
    #         i = 0
    #         while i < len(mselist):
    #             f.write(str(mselist[i]) + '\n')
    #             i += 1
    return mae, mse
