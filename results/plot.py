import os
def makecolor():
    color=['b','g','r','c','m','y','k','b--','g--','r--','c--','m--','y--','k--', 'bo','go','ro','co','mo','yo','ko']
    for i in color:
        yield i
files = [f for f in os.listdir('.')]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--files',nargs='+')
args = parser.parse_args()
files=args.files


#files = ['MNIST_LL_Entropy_100.txt','MNIST_lrl_Entropy_100.txt']
make = makecolor()
for file in files:
    f=open(file,"r")
    # print(f.read())
    text = f.read()
    # print(text)
    loc = text.find('\n',1)
    num = text[0:loc]
    #print(num)
    info = text[loc+10:]
    info = info.replace('=','":').replace('(','{"').replace(', ',', "').replace(')','}')
    import json
    info = info.replace("'","\"").replace("Tr","tr").replace("Fa",'fa')
    info = json.loads(info)
    name=''
    name+=info['data']
    if info['rule'] == 'LL':
        name+='_LL'
    elif info['rule'] == 'lrlonly':
        name+='_lrlonly'
    elif info['rule'] == 'lrlonlywsoftmax':
        name+='_lrlonlywsoftmax'
    elif info['rule'] == 'lplwsoftmax':
        name+='_lplwsoftmax'
    elif info['rule'] == 'lpl':
        name+='_lpl'
    try:
        if info['lamb1'] == 0:
            name+='_lamb10'
    except:
        pass
    if 'lpl' in info.keys():
        name+='_lpl'
    elif info['lrl']==True:
        name+='_lrl'
    
    if info['triplet'] == True:
        name+='_triplet'
    if info['tripletlog'] == True:
        name+='log'
    if info['tripletratio'] == True:
        name+='ratio'
    if info['liftedstructured'] == True:
        name+='_ls'
    
    if info['Ltriplet'] == True:
        name+='_Ltriplet'
    if info['Ltripletlog'] == True:
        name+='_Ltripletlog'
    if info['Ltripletratio'] == True:
        name+='_Ltripletratio'
    if info['Lliftedstructured'] == True:
        name+='_Lls'

    if info['rule'] in ['Entropy','Margin']:
        name+='_'+nfo['rule']
    if info['subset'] !=10000:
        name+='_p'+str(info['subset'])
    if info['lamb2'] != 1.:
        name+='_lamb2'+str(int(info['lamb2']))
    if info['lrlbatch'] != 128:
        name+='_lrlbatch'+str(info['lrlbatch'])
    print(name+' = np.array('+num+')')
    print('plt.plot(axis1000[:10],np.mean('+name+',axis=0)[:10],'+'"'+next(make)+'"'+', label = "'+name+'")')#,sep='')


# import matplotlib.pyplot as plt
# plt.plot(x_axis,collect['R3D.txt']['train']['loss'],'b',legend='R3D train loss')
# plt.show()
