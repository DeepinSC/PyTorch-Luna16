import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize

path = '../../data/out/trainMasks.npy'

data = np.load(path)
count = data.shape[0]

# f, plots = plt.subplots(count, 1, figsize=(5, 40))

for i in range(10):
    image = imresize(data[i][0],(128,128))
    plt.axis('off')
    plt.imshow(image)
    plt.show()





