import tensorflow as tf


#### loss to maximize inter cluster distance and minimize intra cluster variance
#
#   X [nums_points, dim]
#   y [num_points]  categorical y
#

def MaxInterMinInner_Add_loss(X, y_cat, alpha=0.5):
    '''
    Max Inter Class distance and Min Innter Class Variance Loss
    :param X: feature embedding [nums_points, dim]
    :param y_cat: ground-truth [num_points]
    :param alpha: weight for inter class distance
    :return:
    '''
    max_num_cluster = tf.reduce_max(y_cat)
    min_dist_miu_ij = 10000 * tf.ones(shape=(), dtype=tf.float32, name='MinimalDistanceBetweenMeans')
    max_var = tf.zeros(shape=(), dtype=tf.float32, name='MaximalVariance')
    max_miu_i = -1 * tf.ones(shape=(), dtype=tf.int32, name='OptimalMiuIndex_i')
    max_miu_j = -1 * tf.ones(shape=(), dtype=tf.int32, name='OptimalMiuIndex_j')

    def body_i(i, min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j):
        j = i + 1
        y_i_idx = tf.where(tf.equal(y_cat, i))[:, 0]
        X_i = tf.gather(X, y_i_idx, axis=0)
        # X_i = X_ph[y_i_idx,...]
        miu_i, var_i = tf.nn.moments(X_i, axes=0)
        max_var = tf.maximum(max_var, tf.reduce_sum(var_i))

        ## Define j-th loop
        condition2 = lambda i, j, miu_i, min_dist_miu_ij, max_num_cluster, max_miu_i, max_miu_j: \
            tf.less_equal(j, max_num_cluster)

        def body_j(i, j, miu_i, min_dist_miu_ij, max_num_cluster, max_miu_i, max_miu_j):
            # global min_dist_miu_ij

            y_j_idx = tf.where(tf.equal(y_cat, j))[:, 0]
            X_j = tf.gather(X, y_j_idx, axis=0)
            miu_j, var_j = tf.nn.moments(X_j, axes=0)

            dist_miu_ij = tf.einsum('i,i->', miu_i - miu_j, miu_i - miu_j)

            # Find the minimal distance between miu i and j
            def ReturnFcn(dist_miu_ij, i, j):
                return dist_miu_ij, i, j

            result = tf.cond(tf.less(dist_miu_ij, min_dist_miu_ij), lambda: ReturnFcn(dist_miu_ij, i, j),
                             lambda: ReturnFcn(min_dist_miu_ij, max_miu_i, max_miu_j))

            min_dist_miu_ij, max_miu_i, max_miu_j = result

            # if tf.less(dist_miu_ij, min_dist_miu_ij):
            j = j + 1

            return i, j, miu_i, min_dist_miu_ij, max_num_cluster, max_miu_i, max_miu_j

        i, j, miu_i, min_dist_miu_ij, max_num_cluster, max_miu_i, max_miu_j = \
            tf.while_loop(condition2, body_j,
                          loop_vars=[i, j, miu_i, min_dist_miu_ij, max_num_cluster, max_miu_i, max_miu_j])

        i = i + 1

        return i, min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j

    def loss_fcn(min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j):
        i = tf.constant(0)
        condition1 = lambda i, min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j: \
            tf.less_equal(i, max_num_cluster)
        i, min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j = \
            tf.while_loop(condition1, body_i, [i, min_dist_miu_ij, max_var, max_num_cluster, max_miu_i, max_miu_j])

        return min_dist_miu_ij, max_var, max_miu_i, max_miu_j

    min_dist_miu_ij, max_var, max_miu_i, max_miu_j = loss_fcn(min_dist_miu_ij, max_var, max_num_cluster, max_miu_i,
                                                              max_miu_j)

    loss = -alpha * tf.log(min_dist_miu_ij) + (1 - alpha) * tf.log(max_var)
    #
    # # loss = -sum_inter_dist / S
    #
    # return loss, all_inter_dist, S_max

    tf.add_to_collection('max_num_cluster',max_num_cluster)

    return loss, min_dist_miu_ij, max_var, max_miu_i, max_miu_j