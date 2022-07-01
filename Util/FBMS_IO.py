import numpy as np
import os


def ExportPred(Traj,pred,seq_len,filepath):
# function to export prediction in the format defined by FBMS for PR evaluation

    nTraj = pred.shape[0]

    fid = open(filepath,'w')

    fid.write('{:d}\n'.format(seq_len))


    fid.write('{:d}\n'.format(nTraj))


    for t_i in range(0,pred.shape[0]):

        frames = Traj[t_i]['frame']
        x = Traj[t_i]['x']
        y = Traj[t_i]['y']

        fid.write('{:d} {:d}\n'.format(pred[t_i],len(frames)))

        for f_i in range(0,len(frames)):

            fid.write('{:6f} {:6f} {:d}\n'.format(x[f_i],y[f_i],frames[f_i]))


    fid.close()
