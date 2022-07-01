import numpy as np
import tensorflow as tf
import os
import sys
import deepdish
import argparse as arg
import time
import parse
import matplotlib.pyplot as plt
import datetime

sys.path.append(os.path.abspath('../Network'))
sys.path.append(os.path.abspath('../Util'))

import SubspaceNet as model
import Loss
import Tool


def LoadGT(gt_path,seq_name):
    ###### Load GT
    ## Load dat file
    Mask = []
    Frame = []

    dat_filepath = os.path.join(gt_path, '{:s}Def.dat'.format(seq_name))

    fid = open(dat_filepath, 'r')

    content = fid.readlines()

    for l_i in range(0, len(content)):

        line = content[l_i]

        if 'Total number of frames in this shot' in line:
            seq_len = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'Total number of labeled frames' in line:
            num_label_frames = parse.parse('{:d}\n', content[l_i + 1])[0]

        if 'File name' in line:
            ## load the mask file/image
            gt_filepath = os.path.join(gt_path,
                                       parse.parse('{}\n', content[l_i + 1])[0])
            frame_no = parse.parse('{:d}\n', content[l_i - 1])[0] + 1
            img = plt.imread(gt_filepath)
            Mask.append(img)
            Frame.append(frame_no)

    fid.close()

    return Mask, Frame, seq_len


parser = arg.ArgumentParser(description='Take parameters')

parser.add_argument('--GPU',type=int,help='GPU to use',default=0)
parser.add_argument('--Depth',type=int,help='Depth of ResNet',
                    default=50)
parser.add_argument('--Loss',type=str,help='Training Loss',
                    default='L2Loss')
parser.add_argument('--ExpSum',type=int,help='Flag to indicate if export summary',default=0)    # bool
parser.add_argument('--SaveMdl',type=int,help='Flag to indicate if save learned model',default=1)    # bool
parser.add_argument('--LearningRate',type=float,help='Learning Rate',
                    default=1e-2)
parser.add_argument('--Epoch',type=int,help='Number of epochs to train',default=300)
parser.add_argument('--HiddenDim',type=int,help='Hidden dimension',default=256)
parser.add_argument('--Continue',type=int,help='Flag to indicate whether continue training',default=0)


args = parser.parse_args()
Para = {}
Para['GPU'] = args.GPU
Para['depth'] = args.Depth
Para['SUMMARY_LOG_FLAG'] = args.ExpSum
Para['SAVE_MDL_FLAG'] = args.SaveMdl
Para['LearningRate'] = args.LearningRate
Para['Epoch'] = args.Epoch
HiddenDim = args.HiddenDim
ContinueFlag = args.Continue

if Para['GPU'] != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(Para['GPU'])

## Parameters
MaxClust = 10
output_dim = 1*MaxClust
FixedLen = 10 # fixed length of clip
X_dim = 2*FixedLen
y_dim = MaxClust
gap_str = '1'

lossname = args.Loss

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get current time

MODEL_NAME = 'SubspaceNet{:d}_{}'.format(Para['depth'],lossname)
DATA_DIR = os.path.expanduser('../Results/FBMS')
RESULT_DIR = os.path.join(DATA_DIR, '{}_fixedLen-{}_HiddenDim-{:d}_{}'.
                          format(MODEL_NAME, FixedLen, HiddenDim, time))
CHECKPOINT_PATH = os.path.join(RESULT_DIR, 'CheckPoint')
SUMMARY_PATH = os.path.join(RESULT_DIR, 'Summary')
summary_filepath = os.path.join(SUMMARY_PATH, 'Summary.txt')
EMBED_PATH = os.path.join(RESULT_DIR,'Embedding')


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)


seq_tr = 'cars2,cars3,cars6,cars7,cars8,cars9,marple1,marple3,marple5,marple8,marple10,marple11,marple13,bear01,bear02,cats02,cats04,cats05,cats07,ducks01,horses01,horses03,horses06,lion02,meerkats01,people04,people05,rabbits01,rabbits05'
seqlist_tr = seq_tr.split(',')

seq_te = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist_te = seq_te.split(',')



## Define placeholder/inputs
# H Feat Embedding B*N*D
H_ph = tf.placeholder(dtype=tf.float32,shape=[None, None, X_dim],name='inputs/input_feature_embedding')
# y groun-truth labels B*N*C
y_gt_ph = tf.placeholder(dtype=tf.float32,shape=[None, None, y_dim],name='inputs/categorical_label')

Is_Training_ph = tf.placeholder(dtype=tf.bool,shape=[])

## Define GCN network
y_hat = model.SubspaceNet(
    H=H_ph,input_dim=X_dim,output_dim=output_dim,Is_Training=Is_Training_ph,hidden_dim=HiddenDim,depth=Para['depth'])

y_hat = tf.nn.l2_normalize(y_hat, axis=2)

loss = Loss.Loss_L2(y_gt_ph,y_hat)

### Define solver
train_vars = tf.trainable_variables()

## Initialize
solver = tf.train.AdamOptimizer(learning_rate=Para['LearningRate']).minimize(loss)  # initialize solver

saver = tf.train.Saver(max_to_keep=2)

config = tf.ConfigProto(allow_soft_placement=False)
# config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory

sess = tf.Session(config=config)


## Continue Training from previous checkpoint

if ContinueFlag:
    mdl_exist = False
    checkpoint_summary_filepath = os.path.join(CHECKPOINT_PATH,'checkpoint')
    fid = open(checkpoint_summary_filepath,'r')
    contents = fid.readlines()
    for line in contents:
        if 'model_checkpoint_path' in line:
            tmp = parse.parse('{}: "{}"',line)
            ckpt_filepath = tmp[1]
            saver.restore(sess,ckpt_filepath)
            print('Successfully resotred {:s}'.format(ckpt_filepath))
            mdl_exist = True
            break
    fid.close()
    if mdl_exist:
        epoch_start = parse.parse('model_epoch-{:d}.ckpt',ckpt_filepath.split('/')[-1])[0]
    else:
        print('Failed to restore any model. Train from scratch.')
        sess.run(tf.global_variables_initializer())
        epoch_start = 0
else:
    sess.run(tf.global_variables_initializer())
    epoch_start = 0

### Write into summary
if Para['SUMMARY_LOG_FLAG']:
    fid = open(summary_filepath, 'a+')


### Start Training
best_te_acc = 0.

num_epoch = Para['Epoch']

for epoch in range(epoch_start, num_epoch):

    # Train on current clip
    loss_tr_epoch = 0.

    # Random shuffle sequence
    seqlist_tr_epoch = seqlist_tr
    np.random.shuffle(seqlist_tr_epoch)

    for seq_name in seqlist_tr_epoch:

        ## Directories
        base_path = os.path.abspath(
            '../Dataset/FBMS/Data/{}'.format(seq_name))
        gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
        seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
        newclip_path = base_path

        ## Load GT
        Mask, LabelledFrames, seq_len = LoadGT(gt_path, seq_name)

        ## Train each clip
        # random shuffle clips
        clips = list(range(0, len(LabelledFrames)))
        np.random.shuffle(clips)

        for clip_i in clips:

            with tf.device('/cpu:0'):
                # Load New Clip Data
                newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                                format(clip_i, FixedLen, gap_str))

                Data = deepdish.io.load(newclip_filepath)['Data']

                Px = np.array(Data['x'])
                Py = np.array(Data['y'])
                y_cat = np.array(Data['label'])

                X_tr = np.concatenate([Px, Py], axis=1)
                y_onehot_tr = np.zeros(shape=[y_cat.shape[0],10])
                for i in range(y_cat.shape[0]):
                    y_onehot_tr[i,y_cat[i]] = 1

            # Train one batch
            _,loss_mb = sess.run(
            [solver,loss],
            feed_dict={H_ph: np.expand_dims(X_tr, axis=0), y_gt_ph: np.expand_dims(y_onehot_tr,axis=0),
                       Is_Training_ph: True})

            loss_tr_epoch += loss_mb

            print('\rTrained epoch-{:d} seq-{:s} clip-{:d}'.format(epoch,seq_name,clip_i),end='')

    if epoch % 10 == 0:


        ### Evaluate each training example one by one
        acc_tr_all = []
        pred_tr_all = []
        loss_tr_all = 0

        for seq_name in seqlist_tr:
            ## Directories
            base_path = os.path.abspath(
                '../Dataset/FBMS/Data/{}'.format(seq_name))
            gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
            seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
            newclip_path = base_path

            ## Load GT
            Mask, LabelledFrames, seq_len = LoadGT(gt_path, seq_name)

            ## Evaluate 1st clip of each training sequence
            for clip_i in range(0, 1):

                with tf.device('/cpu:0'):
                    # Load New Clip Data
                    newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                                    format(clip_i, FixedLen, gap_str))

                    Data = deepdish.io.load(newclip_filepath)['Data']

                    Px = np.array(Data['x'])
                    Py = np.array(Data['y'])
                    y_cat = np.array(Data['label'])

                    ## Skip if less than 20 points
                    if Px.shape[0] < 20:
                        continue

                    try: X_tr = np.concatenate([Px, Py], axis=1)
                    except:
                        a=1
                    y_onehot_tr = np.zeros(shape=[y_cat.shape[0], 10])
                    for i in range(y_cat.shape[0]):
                        y_onehot_tr[i, y_cat[i]] = 1


                loss_tr, y_hat_tr = sess.run(
                    [loss, y_hat],
                    feed_dict={H_ph: np.expand_dims(X_tr, axis=0), y_gt_ph: np.expand_dims(y_onehot_tr,axis=0),
                               Is_Training_ph: False})

                loss_tr_all += loss_tr

                Nc = len(np.unique(y_cat))
                pred_tr = Tool.Kmeans_Inference(y_hat_tr[0, ...], Nc)
                # A = np.einsum('ij,kj->ik', y_hat_te[0, ...], y_hat_te[0, ...])
                # pred_te = Tool.SpectralClustering(K_hat_te[0,...], Nc)
                # y_gt = np.argmax(y_cat, axis=2)[0, ...]
                acc_tr = Tool.EvalAcc_Fast(pred_tr, y_cat)

                acc_tr_all.append(acc_tr)
                pred_tr_all.append(pred_tr)

                print('\rtrain seq-{:s} acc-{:.2f}'.format(seq_name,acc_tr),end='')


        ### Evaluate each testing example one by one

        acc_te_all = []
        pred_te_all = []
        loss_te_all = 0

        for seq_name in seqlist_te:
            ## Directories
            base_path = os.path.abspath(
                '../Dataset/FBMS/Data/{}'.format(seq_name))
            gt_path = os.path.abspath('../Dataset/FBMS/Data/{:s}'.format(seq_name))
            seq_path = os.path.abspath('../Dataset/FBMS/Data/{:s}/Seq'.format(seq_name))
            newclip_path = base_path

            ## Load GT
            Mask, LabelledFrames, seq_len = LoadGT(gt_path, seq_name)

            ## Evaluate 1st clip of each testing sequence
            for clip_i in range(0, 1):

                with tf.device('/cpu:0'):
                    # Load New Clip Data
                    newclip_filepath = os.path.join(newclip_path, 'Clip-{:d}_nframe-{:d}_gap-{:s}.h5'.
                                                    format(clip_i, FixedLen, gap_str))

                    Data = deepdish.io.load(newclip_filepath)['Data']

                    Px = np.array(Data['x'])
                    Py = np.array(Data['y'])
                    y_cat = np.array(Data['label'])

                    ## Skip if less than 20 points
                    if Px.shape[0] < 20:
                        continue

                    try: X_te = np.concatenate([Px, Py], axis=1)
                    except:
                        a=1
                    y_onehot_te = np.zeros(shape=[y_cat.shape[0], 10])
                    for i in range(y_cat.shape[0]):
                        y_onehot_te[i, y_cat[i]] = 1


                loss_te, y_hat_te = sess.run(
                    [loss, y_hat],
                    feed_dict={H_ph: np.expand_dims(X_te, axis=0), y_gt_ph: np.expand_dims(y_onehot_te,axis=0),
                               Is_Training_ph: False})

                loss_te_all += loss_te

                Nc = len(np.unique(y_cat))
                pred_te = Tool.Kmeans_Inference(y_hat_te[0, ...], Nc)
                # A = np.einsum('ij,kj->ik', y_hat_te[0, ...], y_hat_te[0, ...])
                # pred_te = Tool.SpectralClustering(K_hat_te[0,...], Nc)
                # y_gt = np.argmax(y_cat, axis=2)[0, ...]
                acc_te = Tool.EvalAcc_Fast(pred_te, y_cat)

                acc_te_all.append(acc_te)
                pred_te_all.append(pred_te)

                print('\rtest seq-{:s} acc-{:.2f}'.format(seq_name,acc_te),end='')

        # print('Epoch - {:d} training loss - {:.3f} acc - {:.2f}%\n'.
        #       format(epoch,loss_tr_epoch,100*np.mean(acc_tr_epoch)))
        print('\nEpoch - {:d}\n training loss - {:.2f} acc - {:.2f}%\n'
              'testing loss - {:.2f} acc - {:.2f}%\n'.
              format(epoch,loss_tr_all, 100 * np.mean(acc_tr_all), loss_te_all, 100 * np.mean(acc_te_all)))

        # Save Summary
        if Para['SUMMARY_LOG_FLAG']:
            fid.write('\nEpoch - {:d}\n training loss - {:.2f} acc - {:.2f}%\n'
              'testing loss - {:.2f} acc - {:.2f}%\n'.
              format(epoch,loss_tr_all, 100 * np.mean(acc_tr_all), loss_te_all, 100 * np.mean(acc_te_all)))

            # Write time
            t_str = time.asctime(time.localtime(time.time()))
            fid.write('Finished at: {}\n\n'.format(t_str))


        # Save Checkpoint
        if Para['SAVE_MDL_FLAG']:
            save_path = saver.save(sess, os.path.join(CHECKPOINT_PATH,
                                                  "model_epoch-{:d}.ckpt".format(epoch)))


        ## Update the Best Model
        if best_te_acc < np.mean(acc_te_all):
            best_te_acc = np.mean(acc_te_all)

            ## Preserve best model
            command = 'cp {:s} {:s}'.format(os.path.join(CHECKPOINT_PATH,
                                                "model_epoch-{:d}.ckpt.data-00000-of-00001".format(epoch)),
                                            os.path.join(CHECKPOINT_PATH,
                                                "model_epoch-{:s}.ckpt.data-00000-of-00001".format('best')))
            os.system(command)

            command = 'cp {:s} {:s}'.format(os.path.join(CHECKPOINT_PATH,
                                                         "model_epoch-{:d}.ckpt.index".format(epoch)),
                                            os.path.join(CHECKPOINT_PATH,
                                                         "model_epoch-{:s}.ckpt.index".format('best')))
            os.system(command)
            command = 'cp {:s} {:s}'.format(os.path.join(CHECKPOINT_PATH,
                                                         "model_epoch-{:d}.ckpt.meta".format(epoch)),
                                            os.path.join(CHECKPOINT_PATH,
                                                         "model_epoch-{:s}.ckpt.meta".format('best')))
            os.system(command)