import tensorflow as tf
import sys
import os
import socket
import datetime
import pathlib
import scipy.io as scio
import numpy as np

sys.path.append(os.path.join(str(pathlib.Path(__file__).parent.absolute()),'../Network'))
sys.path.append(os.path.join(str(pathlib.Path(__file__).parent.absolute()),'../Util'))

import Tool
import SubspaceNet as Net
import Loss


class Trainer():

    def __init__(self,args):
        '''
        Initialize
        :param args: Run arguments
        '''

        ## Experiment Settings
        self.Setting = {}
        self.Setting['GPU'] = args.GPU
        self.Setting['split'] = args.Split
        self.Setting['depth'] = args.Depth
        # self.Setting['SUMMARY_LOG_FLAG'] = args.ExpSum
        self.Setting['SAVE_MDL_FLAG'] = args.SaveMdl
        self.Setting['LearningRate'] = args.LearningRate
        self.Setting['Epoch'] = args.Epoch
        self.Setting['HiddenDim'] = args.HiddenDim
        self.Setting['Dataset'] = 'KT3DMoSeg'

        if self.Setting['GPU'] != -1:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.Setting['GPU'])

        ## Model Hyper-parameters
        self.Setting['MaxClust'] = 5    # a possible maximal number of clusters (inference can be more)
        self.Setting['output_dim'] = 1 * self.Setting['MaxClust']   # output feature embedding dimension
        self.Setting['FixedLen'] = 5    # video clip length (frames)
        self.Setting['X_dim'] = 2 * self.Setting['FixedLen']
        self.Setting['Y_dim'] = self.Setting['MaxClust']
        self.Setting['nCV'] = 22  # number of cross validation folds
        self.Setting['lossname'] = 'MaxInterMinIntra'

        ## Dataset
        self.Dataset = {}
        Base_Path = os.path.abspath(os.path.join(str(pathlib.Path(__file__).parent.absolute()),'../Dataset/KT3DMoSeg'))
        self.Dataset['SeqList'] = self.LoadDatasetInfo(Base_Path)


    def LoadDatasetInfo(self, Base_Path):
        '''
        Load KT3DMoSeg Dataset Info
        :param DatasetInfo_Filepath:
        :param Base_Path:
        :param FixedLen:
        :return:
        '''
        DatasetInfo_Filepath = os.path.join(Base_Path, 'SeqList.mat')
        BaseData_Path = os.path.join(Base_Path, 'Data')
        BaseSeq_Path = os.path.join(Base_Path, 'Seq')

        SeqList_py = {}

        temp = scio.loadmat(DatasetInfo_Filepath, struct_as_record=False)

        SeqList = temp['SeqList'][0]

        num_ins = int(SeqList.shape[0])

        for i_i in range(0, num_ins):
            # Load GroundTruth data
            gt_filepath = os.path.expanduser(
                os.path.join(BaseData_Path, '{}_n-{:d}.mat'.format(SeqList[i_i][0], self.Setting['FixedLen'])))
            # temp = scio.loadmat(gt_filepath)

            seq_filepath = os.path.expanduser(os.path.join(BaseSeq_Path,
                                                           '{}'.format(SeqList[i_i][0]), '{:06d}.png'.format(1)))

            SeqList_py[i_i] = {'name': SeqList[i_i][0],
                               'gt_filepath_linux': gt_filepath,
                               'img_path_linux': seq_filepath,
                               'gt_path_linux': BaseData_Path
                               }
        return SeqList_py

    def LoadData(self,gt_filepath):
        '''
        Load data
        :return:
        '''

        temp = scio.loadmat(gt_filepath)

        Data = temp['Data']

        X = Data[0][0][0]
        gt = Data[0][0][1]

        gt = gt - np.min(gt)

        return X, gt

    def PrepareSaveResults(self,save_path=None):
        '''
        Create Saving Paths
        :return:
        '''

        self.Path = {}

        if save_path is None:
            base_path = os.path.abspath(os.path.join(str(pathlib.Path(__file__).parent.absolute()),'../Results'))
            self.Path['result_base_path'] = os.path.join(base_path, self.Setting['Dataset'])
            if not os.path.exists(self.Path['result_base_path']):
                os.makedirs(self.Path['result_base_path'])
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get current time
            self.Path['result_path'] = os.path.join(self.Path['result_base_path'], '{}'.
                                            format(time), 'split-{}'.format(self.Setting['split']))  # complete the result path
            os.makedirs(self.Path['result_path'])
        else:
            self.Path['result_path'] = os.path.join(save_path, 'split-{}'.format(self.Setting['split']))

        # summary output path
        self.Path['summary_path'] = os.path.join(self.Path['result_path'], 'summary')
        os.makedirs(self.Path['summary_path'])
        self.Opt['train_summary_writer'] = tf.summary.FileWriter(os.path.join(self.Path['summary_path'],'train')
                                                           ,self.Opt['sess'].graph)
        self.Opt['valid_summary_writer'] = tf.summary.FileWriter(os.path.join(self.Path['summary_path'],'valid'))

        # checkpoint saving path
        self.Path['ckpt_path'] = os.path.join(self.Path['result_path'], 'ckpt')
        os.makedirs(self.Path['ckpt_path'])
        # feature embedding saving path
        self.Path['embed_path'] = os.path.join(self.Path['result_path'], 'embedding')
        os.makedirs(self.Path['embed_path'])
        # setting file
        self.Path['settings'] = os.path.join(self.Path['result_path'],'settings')
        # results file
        self.Path['results'] = os.path.join(self.Path['result_path'],'results')

    def SaveAllSettings(self):
        '''
        Export experiment settings
        :return:
        '''

        with open(self.Path['settings'],'w') as fid:
            fid.write('Host:{}\n'.format(socket.gethostname()))
            fid.write('GPU:{}\n'.format(self.Setting['GPU']))
            fid.write('LearningRate:{}\n'.format(self.Setting['LearningRate']))
            fid.write('Epoch:{}\n'.format(self.Setting['Epoch']))
            fid.write('loss:{}\n'.format(self.Setting['lossname']))
            fid.write('Target Dataset:{}\n'.format(self.Setting['Dataset']))
            fid.write('NetworkDepth:{}\n'.format(self.Setting['depth']))
            fid.write('FixedLen:{}\n'.format(self.Setting['FixedLen']))
            fid.write('#CrossValid:{}\n'.format(self.Setting['nCV']))
            fid.write('HiddenDim:{}\n'.format(self.Setting['HiddenDim']))

    def DefineNetwork(self):
        '''
        Define Backbone Network
        :return:
        '''

        ## Define placeholder/inputs
        self.Inputs = {}
        self.Outputs = {}
        self.Opt = {}

        # X Input Feature B*N*D
        self.Inputs['X_ph'] = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Setting['X_dim']],
                                             name='inputs/input_feature_embedding')
        # Y Groun-truth labels B*N*C
        self.Inputs['Y_ph'] = tf.placeholder(dtype=tf.float32, shape=[None, None, self.Setting['Y_dim']],
                                             name='inputs/categorical_label')
        # Convert Onehot Key to Category
        self.Outputs['Y_cat_tf'] = tf.cast(tf.argmax(self.Inputs['Y_ph'][0, ...], axis=1), dtype=tf.int32)

        # Is training indicator
        self.Inputs['Is_Training_ph'] = tf.placeholder(dtype=tf.bool, shape=[])

        ## Define backbone network
        Y_hat = Net.SubspaceNet(
            H=self.Inputs['X_ph'],input_dim=self.Setting['X_dim'], output_dim=self.Setting['output_dim'],
            Is_Training=self.Inputs['Is_Training_ph'],hidden_dim=self.Setting['HiddenDim'],depth=self.Setting['depth'])

        self.Outputs['Y_hat'] = tf.nn.l2_normalize(Y_hat, axis=2)

        ## Define Loss
        self.Opt['loss'], self.Opt['loss_inter'], self.Opt['loss_intra'], miu_i, miu_j = Loss.MaxInterMinInner_Add_loss \
                (self.Outputs['Y_hat'][0, ...], self.Outputs['Y_cat_tf'], alpha=0.6)

        ## Add Summary
        tf.summary.scalar('loss',self.Opt['loss'])
        self.Opt['summary'] = tf.summary.merge_all()

    def DefineOptimizer(self):
        '''
        Define Optimizer and Intialize the Network
        :return:
        '''
        ### Define solver

        ## Initialize
        self.Opt['step'] = tf.Variable(0)
        self.Opt['epoch_cnt'] = 0
        self.Opt['solver'] = tf.train.AdamOptimizer(learning_rate=self.Setting['LearningRate']).\
            minimize(self.Opt['loss'], global_step=self.Opt['step'])  # initialize solver

        self.Opt['saver'] = tf.train.Saver(max_to_keep=2)

        config = tf.ConfigProto(allow_soft_placement=False)
        # config=tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth = bool(True)  # Use how much GPU memory

        self.Opt['sess'] = tf.Session(config=config)

        self.Opt['sess'].run(tf.global_variables_initializer())

    def ExportTensorboard(self, writer, name, data):
        '''
        Export train/val and inference results to tensorboard file
        :param result_filepath:
        :return:
        '''
        writer.add_scalar(name, data)

    def DataAugment(self,X,stddev=0.1):
        '''
        Apply Data Augmentation to training trajectories
        :param X: Input trajectories 3*F*L
        :return:
        '''

        ## Apply random Gaussian jittering to trajectories
        jitter = stddev*np.random.randn(X.shape[0],X.shape[1],X.shape[2])
        jitter[2,...] = 0   # not applicable to homogeneous coordinate

        return X+jitter


    def TrainOneEpoch(self):
        '''
        Train one epoch
        :return:
        '''

        # Train on current epoch
        loss_tr_epoch = 0.
        MaxClip = 20    # maximal number of clips per sequences

        ### Random Shuffle
        SeqIndex = list(range(0, len(self.Dataset['SeqList'])))
        np.random.shuffle(SeqIndex)

        for samp_i in SeqIndex:
            # Train on all clips
            acc_tr_epoch = []
            loss_tr_epoch = []

            for r in range(0, MaxClip):
                # Skip any clip overlapping with testing clip
                if samp_i == self.Setting['split'] and r <= self.Setting['FixedLen']:
                    continue

                ## Trajectory Feature Load Data
                gt_filepath = os.path.expanduser(os.path.join(self.Dataset['SeqList'][samp_i]['gt_path_linux'],
                                                              '{}_r-{:d}_v-{:d}.mat'.
                                                              format(self.Dataset['SeqList'][samp_i]['name'],
                                                                     r + 1, r + self.Setting['FixedLen'])))
                # Break if reached the last clip
                if not os.path.exists(gt_filepath):
                    break
                # Load Data
                X, Y = self.LoadData(gt_filepath)

                # Apply Data Augment
                # X = self.DataAugment(X)

                # Reshape input data
                X_tr = np.concatenate([X[0, ...], X[1, ...]], axis=1)
                Y_onehot_tr = Tool.Cat2OneHot_wrap(np.expand_dims(Y, axis=0), self.Setting['MaxClust']).astype(float)

                # Run one step
                summary, step, _,loss_mb, Y_hat_mb = self.Opt['sess'].run(
                [self.Opt['summary'],self.Opt['step'],self.Opt['solver'],self.Opt['loss'], self.Outputs['Y_hat']],
                feed_dict={self.Inputs['X_ph']: np.expand_dims(X_tr, axis=0), self.Inputs['Y_ph']: Y_onehot_tr,
                           self.Inputs['Is_Training_ph']: True})

                self.Opt['train_summary_writer'].add_summary(summary, step)

                loss_tr_epoch.append(loss_mb)

                if self.Opt['epoch_cnt'] % 5 == 0:
                    ### Evaluate Training Samples Performance
                    Nc = len(np.unique(Y))
                    pred_tr = Tool.Kmeans_Inference(Y_hat_mb[0, ...], Nc)
                    acc_tr = Tool.EvalAcc_Fast(pred_tr, Y[:, 0])
                    acc_tr_epoch.append(acc_tr)

                print('\rSplit-{} Trained seq-{:d} clip-{:d}'.format(self.Setting['split'],samp_i,r),end='')

            # average loss per epoch
            loss_tr_epoch = np.mean(loss_tr_epoch)

        if self.Opt['epoch_cnt'] % 5 == 0:
            ### Evaluate each validation example one by one

            acc_te_all = []
            pred_te_all = []
            loss_te_epoch = []

            for samp_i in range(0, len(self.Dataset['SeqList'])):

                if samp_i != self.Setting['split']:
                    continue

                # Load Data
                r=1
                gt_filepath = os.path.expanduser(
                    os.path.join(self.Dataset['SeqList'][samp_i]['gt_path_linux'], '{}_r-{:d}_v-{:d}.mat'.
                                 format(self.Dataset['SeqList'][samp_i]['name'], r, r + self.Setting['FixedLen'] - 1)))
                X, Y = self.LoadData(gt_filepath)

                X_te = np.concatenate([X[0, ...], X[1, ...]], axis=1)
                Y_onehot_te = Tool.Cat2OneHot_KT3DMoSeg_wrap(np.expand_dims(Y, axis=0),
                                                             self.Setting['MaxClust']).astype(float)

                summary, step, loss_te, Y_hat_te = self.Opt['sess'].run(
                    [self.Opt['summary'],self.Opt['step'],self.Opt['loss'], self.Outputs['Y_hat']],
                    feed_dict={self.Inputs['X_ph']: np.expand_dims(X_te, axis=0), self.Inputs['Y_ph']: Y_onehot_te,
                               self.Inputs['Is_Training_ph']: False})

                self.Opt['valid_summary_writer'].add_summary(summary, step)

                loss_te_epoch.append(loss_te)

                acc_te, pred_te = Tool.Kmeans_wrap(Y_hat_te, Y_onehot_te)

                acc_te_all.append(acc_te)
                pred_te_all.append(pred_te)

            # average test loss
            loss_te_epoch = np.mean(loss_te_epoch)

            print('\nSplit-{} Epoch - {:d} training loss - {:.3f} acc - {:.2f}%\n'.
                  format(self.Setting['split'], self.Opt['epoch_cnt'], loss_tr_epoch, 100 * np.mean(acc_tr_epoch)))
            print('testing loss - {:.3f} acc - {:.2f}%\n'.
                  format(loss_te_epoch, 100 * np.mean(acc_te_all)))

            ## Export Valid Results
            if self.Setting['SAVE_MDL_FLAG']:
                with open(self.Path['results'], 'a+') as fid:
                    fid.write(
                        'tesplit-{:d} epoch - {:d}\ntraining loss - {:.2f} acc - {:.2f}%\ntesting loss - {:.2f} acc - {:.2f}%\n'.
                            format(self.Setting['split'], self.Opt['epoch_cnt'], loss_tr_epoch,
                                   100 * np.mean(acc_tr_epoch), loss_te_epoch,
                                   100 * np.mean(acc_te_all)))

            ## Save Checkpoint
            self.Opt['saver'].save(self.Opt['sess'], os.path.join(self.Path['ckpt_path'],
                                                      "model_epoch-{:d}.ckpt".format(self.Opt['epoch_cnt'])))

        ## Increment Checkpoint
        self.Opt['epoch_cnt'] += 1
