import matplotlib.pyplot as plt
import numpy as np

path = '../../data/out/trainImages.npy'

data = np.load(path)
count = data.shape[0]

# f, plots = plt.subplots(count, 1, figsize=(5, 40))

for i in range(count):
    image = data[i][0]
    plt.axis('off')
    plt.imshow(image)
    plt.show()





