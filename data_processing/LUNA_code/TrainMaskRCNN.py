import numpy as np
from data_processing.models.maskrcnn.config import Config
import data_processing.models.maskrcnn.utils as utils
import data_processing.models.maskrcnn.visualize as visualize


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
def test_init():

    working_path = "../../data/out/"
    imgs_train = np.load(working_path + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path + "testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + "testMasks.npy").astype(np.float32)

    dataset_train = NodulesDataset()
    dataset_train.load_samples(imgs_train, imgs_mask_train)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NodulesDataset()
    dataset_val.load_samples(imgs_test, imgs_mask_test_true)
    dataset_val.prepare()

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)*255
        mask, class_ids = dataset_train.load_mask(image_id)
        mask = mask * 255
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


test_init()

