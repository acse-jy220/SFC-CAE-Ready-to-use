import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'font.family':'sans-serif'})
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 
mpl.rc('axes', labelsize=12)

if(len(sys.argv) > 1):
      filename = sys.argv[1]
else:
   raise ValueError('please input the txt file for lossplot')

data = np.loadtxt(filename)
n_epochs = data.shape[0]
epochs = np.linspace(1,n_epochs,n_epochs)

plt.figure(figsize=(12,6))
plt.plot(epochs,data[:,0],label='training loss')
plt.plot(epochs,data[:,1],label='validation loss')
plt.yscale('log')
plt.ylabel('losses',  rotation='horizontal')
plt.xlabel('epochs')
plt.legend()
filename = filename.split('.')[:-1]
plt.savefig(''.join(filename) + '.png')
plt.show()