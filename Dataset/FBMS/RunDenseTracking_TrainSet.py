import os
import subprocess
import parse

seq = 'cars2,cars3,cars6,cars7,cars8,cars9,marple1,marple3,marple5,marple8,marple10,marple11,marple13,bear01,bear02,cats02,cats04,cats05,cats07,ducks01,horses01,horses03,horses06,lion02,meerkats01,people04,people05,rabbits01,rabbits05'
seqlist = seq.split(',')

os.chdir('./moseg')

for seq_i in seqlist:

    ## Check computed
    result_path = os.path.join(os.path.expanduser('./Trainingset/Results'),
                               'OchsBroxMalik8_all_0000060.00',seq_i)

    if not os.path.exists(result_path):
        print('start {:s}'.format(seq_i))
        os.makedirs(result_path)
    else:
        print('exist {:s}'.format(seq_i))
        continue

    command = 'ls ./Trainingset/{:s}/{:s}*.jpg | wc -l'.format(seq_i,seq_i)
    # result = subprocess.run(command, stdout=subprocess.PIPE)
    msg = os.popen(command).read()
    nFrames = parse.parse('{:d}\n',msg)[0]

    command = './MoSeg filestructureTrainingSet.cfg {:s} 0 {:d} 8 60'.format(seq_i,nFrames)

    os.system(command)