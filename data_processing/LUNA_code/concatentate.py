import numpy as np

out_path = "../../data/out/"
working_path = "../../data/out0/"
imgs_train = np.load(working_path+"trainImages.npy").astype(np.float32)
imgs_mask_train = np.load(working_path+"trainMasks.npy").astype(np.float32)

imgs_test = np.load(working_path+"testImages.npy").astype(np.float32)
imgs_mask_test_true = np.load(working_path+"testMasks.npy").astype(np.float32)

for i in range(1, 10):
    working_path = "../../data/out" + str(i) + "/"
    train = np.load(working_path + "trainImages.npy").astype(np.float32)
    imgs_train = np.concatenate((imgs_train, train))
    print("imgs_train finished")

np.save(out_path+"trainImages.npy", imgs_train)
del imgs_train

for i in range(1, 10):
    working_path = "../../data/out" + str(i) + "/"
    mask_train = np.load(working_path + "trainMasks.npy").astype(np.float32)
    imgs_mask_train = np.concatenate((imgs_mask_train, mask_train))
    print("imgs_mask_train finished")

np.save(out_path+"trainMasks.npy", imgs_mask_train)
del imgs_mask_train

for i in range(1, 10):
    working_path = "../../data/out" + str(i) + "/"
    test = np.load(working_path + "testImages.npy").astype(np.float32)
    imgs_test = np.concatenate((imgs_test, test))
    print("imgs_test finished")

np.save(out_path+"testImages.npy", imgs_test)
del imgs_test

for i in range(1, 10):
    working_path = "../../data/out" + str(i) + "/"
    mask_test_true = np.load(working_path + "testMasks.npy").astype(np.float32)
    imgs_mask_test_true = np.concatenate((imgs_mask_test_true, mask_test_true))
    print("imgs_mask_test_true finished")
np.save(out_path+"testMasks.npy", imgs_mask_test_true)
del imgs_test

print("process finished")