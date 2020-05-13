import os
files = [f for f in os.listdir('.')]
files = 'val_annotations.txt'

f=open(files,"r")
f=f.read()
f=f.split('\n')
l = len(f)
for i in range(l):
    f[i] = f[i].split('\t')[:2]
f=f[:-1]
for i in range(l):
    if not os.path.isdir(os.path.join('.',f[i][1])):
        os.mkdir(f[i][1])
    os.system('cp ./images/'+f[i][0]+' '+ f[i][1])
# print(f[:3])
