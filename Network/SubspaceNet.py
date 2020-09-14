import tensorflow as tf
import numpy as np
import TF_Tools

from tensorflow.contrib.layers import fully_connected as mlp


def SubspaceNet(H, input_dim, output_dim, Is_Training, hidden_dim=128, depth=30):
    '''
    SubspaceNet
    :param H:  Input feature
    :param input_dim: Input feature dimension
    :param output_dim: Output feature embedding dimension
    :param Is_Training: Indicate if is training
    :param hidden_dim: dimension of hidden feature embedding
    :param depth: depth of network
    :return:
    '''
    ## Layer 1

    H = ContextNorm(H)
    H = mlp_my(H, 4 * hidden_dim, activation_fn=tf.nn.relu, name='mlp1')

    H = ContextNorm(H)
    H = mlp_my(H, hidden_dim, activation_fn=tf.nn.relu, name='mlp2')

    for layer in range(0, depth):
        ### Block Layer N
        Input = H

        ## First Subblock
        # Context Normalization
        H = ContextNorm(H)

        # FC
        H = mlp_my(H, hidden_dim, activation_fn=None, name='mlp_layer{:d}_1'.format(layer))
        H = tf.nn.relu(H)

        ## Second Subblock
        # Context Normalization
        H = ContextNorm(H)

        # FC
        H = mlp_my(H, hidden_dim, activation_fn=None, name='mlp_layer{:d}_2'.format(layer))
        # Activation
        H = tf.nn.relu(H)

        ## ResNet Shortcut Connect
        H += Input

    ## Output Layer
    # Context Normalization
    H = ContextNorm(H)
    # FC
    H = mlp_my(H, output_dim, activation_fn=None, name='mlp_out')

    return H


def ContextNorm(H):
    mean, var = tf.nn.moments(H, axes=1, keep_dims=True)
    H = tf.nn.batch_normalization(H, mean, var, None, None, 1e-3)

    return H


def mlp_my(H, out_dim, activation_fn, name):
    #
    #   H - 1*N*D

    N = H.get_shape()[1]
    D = H.get_shape()[-1]

    W = tf.get_variable(name + '_W', shape=[D, out_dim])
    b = tf.get_variable(name + '_b', shape=[out_dim])

    if activation_fn == None:
        return tf.einsum('ijk,kl->ijl', H, W) + b
    else:
        return activation_fn(tf.einsum('ijk,kl->ijl', H, W) + b)


def Loss_L2(y_gt, y_predict):
    # L2 loss
    #   y_gt - B*N*D

    y_shp = tf.shape(y_gt)
    num_pts = y_shp[1]

    K = tf.matmul(y_gt, tf.transpose(y_gt, perm=[0, 2, 1]))

    loss1 = tf.norm(tf.matmul(y_predict, tf.transpose(y_predict, perm=[0, 2, 1])) - K, axis=[1, 2], ord='fro')

    loss = tf.reduce_mean(loss1)

    return loss


def Loss_CrossEnt(y_gt, y_predict, gamma=0.1):
    ## Cross Entropy Loss

    y_shp = tf.shape(y_gt)
    num_pts = y_shp[1]

    K_gt = tf.matmul(y_gt, tf.transpose(y_gt, perm=[0, 2, 1]))

    K_hat = tf.linalg.einsum('ijk,ilk->ijl', y_predict, y_predict)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=K_gt, logits=K_hat)

    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2]))

    return loss


def Loss_CrossEnt_L2Reg(y_gt, y_predict, gamma=0.1):
    ## Cross Entropy Loss

    y_shp = tf.shape(y_gt)
    num_pts = y_shp[1]

    K_gt = tf.matmul(y_gt, tf.transpose(y_gt, perm=[0, 2, 1]))

    K_hat = tf.linalg.einsum('ijk,ilk->ijl', y_predict, y_predict)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=K_gt, logits=K_hat)

    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2]))

    # L2 Regularization
    vars = tf.trainable_variables()
    regularizers = tf.add_n([tf.nn.l2_loss(i) for i in vars])

    # Total Loss
    loss += gamma * regularizers

    return loss, regularizers


def Loss_SemiHardTriplet(y_gt, y_predict, margin=0.5):
    y_cat_gt = tf.argmax(y_gt, axis=2)

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(y_cat_gt[0, ...], y_predict[0, ...], margin)

    return loss


def Loss_NPair(labels, y_anchor, y_pos, reg_lambda=0.1):
    loss = tf.contrib.losses.metric_learning.npairs_loss(labels, y_anchor, y_pos, reg_lambda=reg_lambda)

    return loss


def Loss_LIFT(y_gt, y_predict, margin=0.5):
    y_cat_gt = tf.argmax(y_gt, axis=2)

    loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(y_cat_gt[0, ...], y_predict[0, ...], margin)

    return loss


def Loss_Clustering(y_gt, y_predict, margin=0.5):
    y_cat_gt = tf.expand_dims(tf.argmax(y_gt, axis=2), axis=2)

    loss = tf.contrib.losses.metric_learning.cluster_loss(y_cat_gt[0, ...], y_predict[0, ...], margin)

    return loss