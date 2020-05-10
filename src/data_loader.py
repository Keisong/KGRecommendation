import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data,data = load_rating(args)
    n_entity, n_relation, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R = load_kg(args)

    dict_temp = {}
    movie_files = '../' + args.dataset + '/item_index_old2new'
    if os.path.exists(movie_files + '.npy'):
        movie_np = np.load(movie_files + '.npy')
    else:
        movie_np = np.loadtxt(movie_files + '.txt', dtype=np.int64)
        np.save(movie_files + '.npy', movie_np)
    print('movie_index loaded.')
    for i in movie_np:
        dict_temp[i[0]] = i[1]
    mname_temp = {}
    file = '../' + args.dataset + '/movies.csv'
    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(',')
        mname_temp[array[0]] = array[1]

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data,adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R, data, dict_temp, mname_temp


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data,data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data,data


def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data,rating_np


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    # consider both output and input degree
    kg = construct_kg(kg_np)
    adj_entity_R, adj_relation_R = construct_adj(args, kg, n_entity, 1)
    adj_entity_N, adj_relation_N = construct_adj(args, kg, n_entity, 2)
    return n_entity, n_relation, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num, m):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    n_neighbor = int(args.neighbor_sample_size/m)
    adj_entity = np.zeros([entity_num, n_neighbor], dtype=np.int64)
    adj_relation = np.zeros([entity_num, n_neighbor], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= n_neighbor:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=n_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=n_neighbor, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
