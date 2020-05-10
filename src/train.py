import tensorflow as tf
import numpy as np
from time import time
from model import KGCN


def train(args, data, show_loss, show_topk, toTrain):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R = data[7], data[8], data[9], data[10]
    data, movie_new2old, mname = data[11], data[12], data[13]

    model = KGCN(args, n_user, n_entity, n_relation, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)
    user_lists, train_records, item_sets = topk_setting(show_topk, data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if toTrain:
            for step in range(args.n_epochs):
                # training
                np.random.shuffle(train_data)
                start = 0
                # skip the last incomplete minibatch if its size < batch size
                while start + args.batch_size <= train_data.shape[0]:
                    _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(start, loss)

                # CTR evaluation
                train_auc, train_f1, train_acc = ctr_eval(sess, model, train_data, args.batch_size)
                eval_auc, eval_f1, eval_acc = ctr_eval(sess, model, eval_data, args.batch_size)
                test_auc, test_f1, test_acc = ctr_eval(sess, model, test_data, args.batch_size)

                print(
                    'epoch %d    train auc: %.4f  acc: %.4f    f1: %.4f    eval auc: %.4f  acc: %.4f    f1: %.4f    test auc: %.4f  acc: %.4f    f1: %.4f'
                    % (step, train_auc, train_acc, train_f1, eval_auc, eval_acc, eval_f1, test_auc, test_acc, test_f1))

                # top-K evaluation
                if show_topk:
                    precision, recall = topk_eval(
                        sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                    print('precision: ', end='')
                    for i in precision:
                        print('%.4f\t' % i, end='')
                    print()
                    print('recall: ', end='')
                    for i in recall:
                        print('%.4f\t' % i, end='')
                    print('\n')
            saver = tf.train.Saver()
            saver.save(sess, "KGCNmodel.ckpt")
        else:
            saver = tf.train.Saver()
            saver.restore(sess, "KGCNmodel.ckpt")
            t = time()
            print("prediction start------------------------")
            user = 13112
            movies = topk_pred(sess, model, user, train_records, item_sets, args.batch_size)
            result = index2movie(mname,movies,movie_new2old)
            print(result)
            print('time used: %d s' % (time() - t))


def index2movie(mname, movies, movie_new2old):
    result = []
    for i in movies:
        result.append(mname[str(movie_new2old[i])])
    return result


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def topk_setting(show_topk, data, n_item):
    if show_topk:
        train_record = get_user_record((data), True)
        user_list = list(set(train_record.keys()))
        item_set = set(list(range(n_item)))
        return user_list, train_record, item_set
    else:
        return [None] * 5


def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, acc = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(acc_list))


def topk_pred(sess, model, user, train_record, item_set, batch_size):
    test_item_list = list(item_set - train_record[user])
    item_score_map = dict()
    start = 0
    while start + batch_size <= len(test_item_list):
        items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                model.item_indices: test_item_list[start:start + batch_size]})
        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += batch_size

    # padding the last incomplete minibatch if exists
    if start < len(test_item_list):
        items, scores = model.get_scores(
            sess, {model.user_indices: [user] * batch_size,
                   model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                           batch_size - len(test_item_list) + start)})
        for item, score in zip(items, scores):
            item_score_map[item] = score

    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    item_sorted = [i[0] for i in item_score_pair_sorted]

    return item_sorted[:10]


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
