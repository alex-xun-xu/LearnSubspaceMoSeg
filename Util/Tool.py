import numpy as np
import sklearn.cluster as cluster
from sklearn.cluster import SpectralClustering as SP

eps_f32 = np.spacing(np.float32(1))


def SpectralClustering(K,n):

    labels = SP(n_clusters=n,affinity='precomputed').fit_predict(K)

    return labels


def Permutations(cand):

    perms = np.empty(shape=[0,cand.shape[0]],dtype=cand.dtype)

    def heapPermutation(cand,size,n,perms):

        if size == 1:
            perms = np.concatenate([perms,np.expand_dims(cand,0)],axis=0)
            return perms

        for i in range(0,size):

            perms = heapPermutation(cand,size-1,n,perms)

            if size % 2 == 1:
                temp = cand[size-1]
                cand[size-1] = cand[0]
                cand[0] = temp

            else:
                temp = cand[size - 1]
                cand[size - 1] = cand[i]
                cand[i] = temp

        return perms

    n = cand.shape[0]
    perms = heapPermutation(cand, n, n, perms)

    return perms


def EvalClassAcc(pred_onehot,gt_onehot):
    ## Evaluate accuracy

    maxout = np.shape(gt_onehot)[2]

    # Generate Permutations
    perms = Permutations(np.arange(0, maxout, dtype=int))
    # 1 to 1
    acc = np.zeros(shape=[gt_onehot.shape[0]])

    for i_i in range(0, gt_onehot.shape[0]):
        for p_i in range(0, int(perms.shape[0])):
            pred_cat = np.argmax(pred_onehot[i_i, ...], axis=1)
            gt_cat = np.argmax(np.einsum('ij->ji', gt_onehot[i_i, :, perms[p_i, :]]), axis=1)
            acc[i_i] = np.max([acc[i_i], np.mean(np.abs(pred_cat - gt_cat) < 0.1)])

    return acc


def EvalAcc(gt_onehot,pred_onehot):
    ## Evaluate accuracy

    maxout = np.shape(gt_onehot)[2]

    # Generate Permutations
    perms = Permutations(np.arange(0, maxout, dtype=int))
    # 1 to 1
    acc = np.zeros(shape=[gt_onehot.shape[0]])

    for i_i in range(0, gt_onehot.shape[0]):
        for p_i in range(0, int(perms.shape[0])):
            pred_cat = np.argmax(pred_onehot[i_i, ...], axis=1)
            gt_cat = np.argmax(np.einsum('ij->ji',gt_onehot[i_i, :, perms[p_i, :]]), axis=1)
            acc[i_i] = np.max([acc[i_i], np.mean(np.abs(pred_cat - gt_cat) < 0.1)])

    return acc

def EvalAcc_Fast(label1,label2):
    ## Evaluate accuracy
    #
    #   label1 - N
    #   label2 - N

    N = label1.shape[0]

    ## Generate Compact Cluster Labels
    Label1_Unique = np.unique(label1)
    Label2_Unique = np.unique(label2)
    label1new = label1
    label2new = label2

    for i in range(0,len(Label1_Unique)):
        label1new[np.where(label1==Label1_Unique[i])[0]] = i

    for i in range(0,len(Label2_Unique)):
        label2new[np.where(label2==Label2_Unique[i])[0]] = i

    label1_onehot = np.zeros([N,len(Label1_Unique)])
    label2_onehot = np.zeros([N, len(Label2_Unique)])

    for i in range(0,N):
        label1_onehot[i,label1new[i]] = 1

    for i in range(0,N):
        label2_onehot[i,label2new[i]] = 1

    index = np.argmin([np.shape(label1_onehot)[1],np.shape(label2_onehot)[1]])
    dim = np.min([np.shape(label1_onehot)[1], np.shape(label2_onehot)[1]])

    # Generate Permutations
    perms = Permutations(np.arange(0, dim, dtype=int))

    if index == 0:
        # Permutate label 1
        acc = 0.
        for p_i in range(0, int(perms.shape[0])):
            label2_cat = np.argmax(label2_onehot, axis=1)
            label1_cat = np.argmax(label1_onehot[:, perms[p_i, :]], axis=1)
            acc = np.max([acc, np.mean(np.abs(label1_cat - label2_cat) < 0.1)])

    else:

        # Permutate label 2
        acc = 0.
        for p_i in range(0, int(perms.shape[0])):
            label1_cat = np.argmax(label1_onehot, axis=1)
            label2_cat = np.argmax(label2_onehot[:, perms[p_i, :]], axis=1)
            acc = np.max([acc, np.mean(np.abs(label1_cat - label2_cat) < 0.1)])

    return acc


def EvalAccSgl(gt_onehot,pred_onehot):
    ## Evaluate accuracy

    maxout = np.shape(gt_onehot)[1]

    # Generate Permutations
    perms = Permutations(np.arange(0, maxout, dtype=int))
    # 1 to 1
    acc = 0.

    for p_i in range(0, int(perms.shape[0])):
        pred_cat = np.argmax(pred_onehot, axis=1)
        gt_cat = np.argmax(gt_onehot[:, perms[p_i, :]], axis=1)
        acc = np.max([acc, np.mean(np.abs(pred_cat - gt_cat) < 0.1)])

    return acc



def pdist2_tf(x,y):
    import tensorflow as tf
    r = tf.reduce_sum(tf.einsum('ijk,ijk->ijk', x, y), axis=2)
    # turn r into column vector
    r = tf.expand_dims(r, axis=2)
    D = r - 2 * tf.einsum('ijk,ilk->ijl', x, y) + tf.einsum('ijk->ikj', r)
    D - tf.sqrt(D)

    return D

## Return binary vector indicating elements of B in A
# return same size of A
def IsMember(A,B):
    VecInd = []
    for a in A:
        if a in B:
            VecInd.append(True)
        else:
            VecInd.append(False)

    return VecInd


def L2Normlize_Batch(x):

    # L2 Normalize
    norm = np.sqrt(np.sum(x ** 2, axis=2))
    x = (x+eps_f32) / np.tile(np.expand_dims(norm+eps_f32, axis=2), [1, 1, x.shape[2]])

    return x

def L2Normlize(x):

    # L2 Normalize
    norm = np.sqrt(np.sum(x ** 2, axis=1))
    x = (x+eps_f32) / np.tile(np.expand_dims(norm+eps_f32, axis=1), [1, x.shape[1]])

    return x


def Kmeans_Inference(x, K):
    import sklearn.cluster as cluster

    num_points = x.shape[0]

    # L2 normalize
    x = L2Normlize(x)

    kmeans = cluster.KMeans(n_clusters=K).fit(x)

    c_pred = kmeans.labels_

    return c_pred

def Kmeans_Inference_mdlsel(x, K):
    import sklearn.cluster as cluster

    num_points = x.shape[0]

    # L2 normalize
    x = L2Normlize(x)

    kmeans = cluster.KMeans(n_clusters=K).fit(x)

    c_pred = kmeans.labels_
    sumdist = kmeans.inertia_

    return c_pred, sumdist


def OnehotEncode(Y,K):
    "Onehot key encoding of input label Y"
    #
    #   Y ~ B*N
    #   K ~ 1   The largest category index
    #

    Y_onehot = np.zeros(shape=[Y.shape[0],Y.shape[1],K])

    for b in range(Y.shape[0]):
        for r in range(Y.shape[1]):

            Y_onehot[b,r,Y[b,r]] = 1

    return Y_onehot



## Test Code

label1_cat = np.argmax(np.random.random([100,5]),axis=1)
label2_cat = np.argmax(np.random.random([100,10]),axis=1)

label1_onehot = np.zeros([100,5])
label2_onehot = np.zeros([100,10])

for i in range(0,100):
    label1_onehot[i,label1_cat[i]] = 1
    label2_onehot[i, label2_cat[i]] = 1

label2_cat = label1_cat
label2_onehot = label1_onehot

acc1 = EvalAccSgl(label1_onehot,label2_onehot)

acc2 = EvalAcc_Fast(label1_cat,label2_cat)

a=1


## Deep Spectral Clustering Inference
#
#   Folow the Algo. 2 in the paper Deep Spectral Clustering Learning, ICML17
#
#   F ~ N*D   : Output features for clustering
#   Nc ~ 1 scalar : Number of clusters
def DeepSpecClustInfer(F, Nc):

    M = F - np.mean(F,axis=0,keepdims=True)
    r = np.linalg.matrix_rank(M)
    U,S,V = np.linalg.svd(M)
    U = U[:,1:r]
    U = L2Normlize(U)

    return Kmeans_Inference(U, Nc)


#### function to convert categorical label to onehot key encodings
#
#   y_cat [batch num_points 1]  min(y_cat)=0
#
def Cat2OneHot_wrap(y_cat,max_cat=-1):

    num_batch = y_cat.shape[0]
    num_points = y_cat.shape[1]
    unique_cats = np.unique(y_cat)
    num_cat = len(unique_cats)
    if max_cat == -1:
        max_cat = num_cat

        # num_cat = np.max(y_cat).astype(int) + 1

    y_onehot = np.zeros(shape=[num_batch,num_points,max_cat],dtype=np.bool)

    for b_i in range(0,num_batch):
        for c_i in range(0,num_cat):
            y_onehot[b_i,y_cat[b_i,:,0]==unique_cats[c_i],c_i] = 1.
            #y_onehot[b_i, y_cat[b_i, :] == unique_cats[c_i], c_i] = 1.
    return y_onehot


#### function to convert categorical label to onehot key encodings
#
#   y_cat [batch num_points 1]  min(y_cat)=0
#
def Cat2OneHot_KT3DMoSeg_wrap(y_cat,max_cat=-1):

    num_batch = y_cat.shape[0]
    num_points = y_cat.shape[1]
    unique_cats = np.unique(y_cat)
    num_cat = len(unique_cats)
    if max_cat == -1:
        max_cat = num_cat

        # num_cat = np.max(y_cat).astype(int) + 1

    y_onehot = np.zeros(shape=[num_batch,num_points,max_cat],dtype=np.bool)

    for b_i in range(0,num_batch):
        for c_i in range(0,num_cat):
            y_onehot[b_i,y_cat[b_i,:,0]==unique_cats[c_i],c_i] = 1.

    return y_onehot

def Kmeans_wrap(x, gt, maxout=-1, k=-1):

    num_points = x.shape[1]

    # L2 normalize
    x = L2Normlize_Batch(x)

    if maxout == -1:
        maxout = int(gt.shape[2])


    # Apply Kmeans for clustering
    c_pred = np.zeros([x.shape[0], num_points, maxout], dtype=np.float)
    for i_i in range(0, x.shape[0]):
        # Determine the number of cluster
        # if np.sum(gt[i_i,...],axis=0)[2]>0:
        #     k = 3
        # else:
        #     k=2

        if k == -1:
            k = np.sum(np.sum(gt[i_i,...],axis=0)>0)

        kmeans = cluster.KMeans(n_clusters=k).fit(x[i_i, :, :])
        for c_i in range(0, maxout):
            c_pred[i_i, kmeans.labels_ == c_i, c_i] = 1
        # c_pred[i_i, kmeans.labels_ == 1, 1] = 1

    ## Evaluate accuracy

    # Generate Permutations
    perms = Permutations(np.arange(0, maxout, dtype=int))
    # 1 to 1
    acc = np.zeros(shape=[x.shape[0]])

    for i_i in range(0, x.shape[0]):
        for p_i in range(0, int(perms.shape[0])):
            pred_cat = np.argmax(c_pred[i_i, ...],axis=1)
            gt_cat = np.argmax(np.transpose(gt[i_i, :, perms[p_i, :]]),axis=1)
            acc[i_i] = np.max([acc[i_i], np.mean(np.abs(pred_cat - gt_cat)<0.1)])

    return acc, c_pred