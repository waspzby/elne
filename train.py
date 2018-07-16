from __future__ import division
from __future__ import print_function

from optimizer import OptimizerTargetOne, OptimizerTargetTwo
from input_data import load_data
from model import TargetOneModel, TargetTwoModel
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate_target_one', 0.01, 'Initial learning rate for learning target 1.')
flags.DEFINE_float('learning_rate_target_two', 0.01, 'Initial learning rate for learning target 2.')
flags.DEFINE_integer('target_one_epochs', 300, 'Number of epochs to train for learning-target-one model.')
flags.DEFINE_integer('target_two_epochs', 300, 'Number of epochs to train for learning-target-two model.')
flags.DEFINE_integer('hidden1', 1000, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 200, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_integration', 0.5, 'Weight for integration for embeddings of learning-target-two model.')
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')

dataset_str = FLAGS.dataset

# Load data
adj, features, y_train, train_mask, y_all = load_data(dataset_str)

adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

# Spectral transform of adjacency matrix
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'labels':  tf.placeholder(tf.float32),
    'labels_mask': tf.placeholder(tf.int32)
}

num_nodes = adj.shape[0]

features = adj_norm
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create models
target_one_model = TargetOneModel(placeholders, num_features, features_nonzero)
target_two_model = TargetTwoModel(placeholders, num_features, features_nonzero, y_train.shape[1])

# Reweights for terms in learning-target-one model's cross entropy loss
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


def check_multi_label_classification(Y, emb, test_ratio):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape, np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
        for i in range(y_test.shape[0]):
            num = int(sum(y_test[i]))
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new

    x_train, x_test, y_train, y_test = train_test_split(emb, Y, test_size=test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)

    # micro = f1_score(y_test, y_pred, average="micro")
    # macro = f1_score(y_test, y_pred, average="macro")
    # return "micro_f1: %.4f macro_f1 : %.4f" % (micro, macro)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    return "accuracy: %.4f" % accuracy

# Target one optimizer
with tf.name_scope('optimizer_for_target_one'):
    opt = OptimizerTargetOne(preds=target_one_model.reconstructions,
                             labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                             pos_weight=pos_weight,
                             norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

feed_dict = construct_feed_dict(adj_label, features, placeholders)
feed_dict.update({placeholders['dropout']: FLAGS.dropout})

# Train learning-target-one model
for epoch in range(FLAGS.target_one_epochs):

    t = time.time()

    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.2f}".format(avg_cost / 100000),
          "train_acc=", "{:.5f}".format(avg_accuracy), "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished for Learning-target-one model!")

feed_dict.update({placeholders['dropout']: 0})
target_one_emb = sess.run(target_one_model.z_mean, feed_dict=feed_dict)

print('Test Classification (training set 10%) for Learning-target-one model: ' + check_multi_label_classification(y_all, target_one_emb, test_ratio=0.9))

print("----------------------------------------------------------------------------------------------------------")

# Target two optimizer
with tf.name_scope('optimizer_for_target_two'):
    opt = OptimizerTargetTwo(preds=target_two_model.outputs,
                             labels=target_two_model.labels,
                             masks=target_two_model.labels_mask)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict = dict()
feed_dict.update({placeholders['labels']: y_train})
feed_dict.update({placeholders['labels_mask']: train_mask})
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
feed_dict.update({placeholders['features']: features})

# Train learning-target-two model
for epoch in range(FLAGS.target_two_epochs):

    t = time.time()

    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy))

print("Optimization Finished for Learning-target-two model!")

feed_dict.update({placeholders['dropout']: 0})
target_two_emb_1st_hidden_layer = sess.run(target_two_model.hidden1, feed_dict=feed_dict)
print('Test Classification (training set 10%) for 1st hidden layer of Learning-target-two model: ' + check_multi_label_classification(y_all, target_two_emb_1st_hidden_layer, test_ratio=0.9))

feed_dict.update({placeholders['dropout']: 0})
target_two_emb_2nd_hidden_layer = sess.run(target_two_model.embeddings, feed_dict=feed_dict)
print('Test Classification (training set 10%) for 2nd hidden layer(embedding) of Learning-target-two model: ' + check_multi_label_classification(y_all, target_two_emb_2nd_hidden_layer, test_ratio=0.9))

print("----------------------------------------------------------------------------------------------------------")

integrated_emb = (1 - FLAGS.weight_integration) * target_one_emb + FLAGS.weight_integration * target_two_emb_2nd_hidden_layer
print('Test Classification (training set 10%) for the integrated embeddings: ' + check_multi_label_classification(y_all, emb=integrated_emb, test_ratio=0.9))
