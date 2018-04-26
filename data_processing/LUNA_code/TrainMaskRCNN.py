import numpy as np
from data_processing.models.maskrcnn.config import Config
import data_processing.models.maskrcnn.utils as utils
import data_processing.models.maskrcnn.visualize as visualize
from data_processing.models.maskrcnn.MaskRCNN import MaskRCNN, load_image_gt,mold_image
import os
from keras import backend as K
import matplotlib.pyplot as plt
from scipy.misc import imresize

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
smooth = 1.
working_path = "../../data/out/"
img_rows = 128
img_cols = 128

def get_data(downsample=True):
    imgs_train = np.load(working_path + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path + "testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + "testMasks.npy").astype(np.float32)

    if downsample:
        temp = np.zeros([imgs_train.shape[0], img_rows, img_cols, imgs_train.shape[1]])
        for i in range(imgs_train.shape[0]):
            temp[i,:,:,0] = imresize(imgs_train[i,:,:,0],(128,128))
        imgs_train = temp

        temp = np.zeros([imgs_mask_train.shape[0], img_rows, img_cols, imgs_mask_train.shape[1]])
        for i in range(imgs_mask_train.shape[0]):
            temp[i,:,:,0] = imresize(imgs_mask_train[i,:,:,0], (128, 128))
        imgs_mask_train = temp

        temp = np.zeros([imgs_test.shape[0], img_rows, img_cols, imgs_test.shape[1]])
        for i in range(imgs_test.shape[0]):
            temp[i,:,:,0] = imresize(imgs_test[i,:,:,0], (128, 128))
        imgs_test = temp

        temp = np.zeros([imgs_mask_test_true.shape[0], img_rows, img_cols, imgs_mask_test_true.shape[1]])
        for i in range(imgs_mask_test_true.shape[0]):
            temp[i,:,:,0] = imresize(imgs_mask_test_true[i,:,:,0], (128, 128))
        imgs_mask_test_true = temp

    return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test_true

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class NodulesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nodules"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2  # batch size

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + nodule

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class NodulesDataset(utils.Dataset):
    '''
    overwrite these functions:
        load_image()
        load_mask()
        image_reference()
    '''
    images = np.array([])
    masks = np.array([])

    def load_samples(self, img_npy, mask_npy):
        if len(img_npy.shape) != 4:
            print("**Input image size should be [num, channel, width, height] **")
            return

        # Add classes
        self.add_class("nodules", 1, "nodules")

        num, channel, width, height = img_npy.shape
        for i in range(img_npy.shape[0]):
            self.add_image("nodules", image_id=i, path=None,
                           width=width, height=height, nodules=['nodules',])
        self.images = img_npy
        self.masks = mask_npy

    def load_image(self, image_id):
        return self.images[image_id]

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        nodules = info['nodules']
        class_ids = np.array([self.class_names.index(n) for n in nodules])
        return self.masks[image_id], class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the nodules data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nodules":
            return info["nodules"]
        else:
            super(self.__class__).image_reference(self, image_id)





# Training dataset
def dataset_init(plot=False, downsample=True):
    print("** Begin to create Nodule Dataset... **")

    imgs_train, imgs_mask_train, imgs_test, imgs_mask_test_true = get_data(downsample)

    dataset_train = NodulesDataset()
    dataset_train.load_samples(imgs_train, imgs_mask_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NodulesDataset()
    dataset_val.load_samples(imgs_test, imgs_mask_test_true)
    dataset_val.prepare()

    if plot:
        image_ids = np.random.choice(dataset_train.image_ids, 4)
        for image_id in image_ids:
            image = dataset_train.load_image(image_id)*255
            mask, class_ids = dataset_train.load_mask(image_id)
            mask = mask * 255
            visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
    print("** Nodule Dataset Ready **")
    return dataset_train, dataset_val


def train_test_model(dataset_train, dataset_val, model_config, out_dir = 'data_processing/models/out/'):
    model = MaskRCNN(mode="training", config=model_config, model_dir=out_dir)
    # pre-train
    model.train(dataset_train,dataset_val,
                learning_rate=model_config.LEARNING_RATE,
                epochs=1,
                layers='heads')
    # fine-tune
    model.train(dataset_train, dataset_val,
                learning_rate=model_config.LEARNING_RATE / 10,
                epochs=1,
                layers="all")


    # test
    class InferenceConfig(NodulesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()
    model = MaskRCNN(mode="inference",
                     config=inference_config,
                     model_dir=MODEL_DIR)

    # TODO: 这里应该用测试集
    # image_ids = np.random.choice(dataset_val.image_ids)
    image_ids = dataset_val.image_ids
    # APs = []
    # Precisions = []
    # Recalls = []
    mean = 0.0
    count = 0
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset_val, inference_config,
                          image_id, use_mini_mask=False)
        mask, class_ids = dataset_val.load_mask(image_id)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        # AP, precisions, recalls, overlaps = \
        #    utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                     r["rois"], r["class_ids"], r["scores"], r['masks'])
        # APs.append(AP)
        # Precisions.append(precisions)
        # Recalls.append(recalls)
        print(r['masks'].shape)
        mean += dice_coef_np(mask, np.mean(r['masks'],axis=2))
        if count == 0:
            plt.imshow(np.mean(r['masks'],axis=2), cmap='gray')
            print(np.mean(r['masks'],axis=2))
            plt.show()
        count += 1
        print(count)
    mean /= len(image_ids)
    print("Mean Dice Coeff : ", mean)

    # print("mAP: ", np.mean(APs))
    # print("mPrecision: ", np.mean(Precisions))
    # print("mRecall: ", np.mean(Recalls))


config = NodulesConfig()
train, val = dataset_init()
# print(val)
train_test_model(train, val, config)


