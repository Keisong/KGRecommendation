import tensorflow as tf
import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R):
        self._parse_args(args, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R)
        self._build_inputs()
        alpha = args.alpha
        beta = args.beta
        self._build_model(n_user, n_entity, n_relation,alpha,beta)
        self._build_train(alpha,beta)

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity_N, adj_relation_N, adj_entity_R, adj_relation_R):
        # [entity_num, neighbor_sample_size]
        self.adj_entity_N = adj_entity_N
        self.adj_relation_N = adj_relation_N
        self.adj_entity_R = adj_entity_R
        self.adj_relation_R = adj_relation_R

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user, n_entity, n_relation, alpha, beta):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        # [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities_N, relations_N = self.get_neighbors(self.item_indices,1)
        entities_R, relations_R = self.get_neighbors(self.item_indices, 2)
        # [batch_size, dim]
        self.item_embeddings_N, self.aggregators_N = self.aggregate(entities_N, relations_N, 1, 2)
        self.item_embeddings_R, self.aggregators_R = self.aggregate(entities_R, relations_R, 2, 1)

        self.item_embeddings = alpha*self.item_embeddings_N + beta*self.item_embeddings_R
        # [batch_size]
        self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds, n):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(int(self.n_iter/n)):   # n_iter means n-layer neighbors, maximum neighbor_sample_size neighbors per layer
            if n == 1:
                neighbor_entities = tf.reshape(tf.gather(self.adj_entity_N, entities[i]),
                                               [self.batch_size, -1])  # gather: extract items ->(items , item_index )
                neighbor_relations = tf.reshape(tf.gather(self.adj_relation_N, entities[i]), [self.batch_size, -1])
            else :
                neighbor_entities = tf.reshape(tf.gather(self.adj_entity_R, entities[i]),
                                               [self.batch_size, -1])  # gather: extract items ->(items , item_index )
                neighbor_relations = tf.reshape(tf.gather(self.adj_relation_R, entities[i]), [self.batch_size, -1])

            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations # Own id and neighbors' id

    def aggregate(self, entities, relations, n, m):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(int(self.n_iter/n)):
            if i == int(self.n_iter/n) - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(int(self.n_iter/n) - i):
                shape = [self.batch_size, -1, int(self.n_neighbor/m), self.dim] # n_neighbor means cut with n neighbor entities
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self, alpha, beta):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss_N = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        self.l2_loss_R = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators_N:
            self.l2_loss_N = self.l2_loss_N + tf.nn.l2_loss(aggregator.weights)
        for aggregator in self.aggregators_R:
            self.l2_loss_R = self.l2_loss_R + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss_N*beta + self.l2_weight*self.l2_loss_R*alpha
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        acc = np.mean(np.equal(scores, labels))
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1, acc

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
