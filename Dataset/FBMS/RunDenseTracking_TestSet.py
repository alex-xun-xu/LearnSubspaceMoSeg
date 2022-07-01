import os
import subprocess
import parse

seq = 'cars1,cars4,cars5,cars10,marple2,marple4,marple6,marple7,marple9,marple12,people1,people2,tennis,camel01,cats01,cats03,cats06,dogs01,dogs02,farm01,goats01,horses02,horses04,horses05,lion01,people03,giraffes01,rabbits02,rabbits03,rabbits04'
seqlist = seq.split(',')

os.chdir('/home/elexuxu/vision/Alex/Data/FBMS/moseg')

for seq_i in seqlist:

    ## Check computed
    result_path = os.path.join(os.path.expanduser('./Testset/Results'),
                               'OchsBroxMalik8_all_0000060.00',seq_i)

    if not os.path.exists(result_path):
        print('start {:s}'.format(seq_i))
        os.makedirs(result_path)
    else:
        print('exist {:s}'.format(seq_i))
        continue

    command = 'ls ./Testset/{:s}/{:s}*.jpg | wc -l'.format(seq_i,seq_i)
    # result = subprocess.run(command, stdout=subprocess.PIPE)
    msg = os.popen(command).read()
    nFrames = parse.parse('{:d}\n',msg)[0]

    command = './MoSeg filestructureTestSet.cfg {:s} 0 {:d} 8 60'.format(seq_i,nFrames)

    os.system(command)