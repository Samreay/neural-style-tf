from scipy.io import savemat, loadmat
import numpy as np

vgg_layers = loadmat('imagenet-vgg-verydeep-19.mat')['layers'][0]
vgg_compressed = []
for i in [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]:
    vgg_compressed.append([vgg_layers[i][0][0][2][0][0], vgg_layers[i][0][0][2][0][1]])
vgg_compressed = np.array(vgg_compressed)
savemat('imagenet-vgg-verydeep-19-compressed.mat', {'layers': vgg_compressed})