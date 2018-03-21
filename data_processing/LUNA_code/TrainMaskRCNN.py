import numpy as np
from data_processing.models.maskrcnn.config import Config
import data_processing.models.maskrcnn.utils as utils
import data_processing.models.maskrcnn.visualize as visualize
from data_processing.models.maskrcnn.MaskRCNN import MaskRCNN, load_image_gt,mold_image


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
    working_path = "../../data/out/"
    imgs_train = np.load(working_path + "trainImages.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainMasks.npy").astype(np.float32)

    imgs_test = np.load(working_path + "testImages.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + "testMasks.npy").astype(np.float32)

    img_rows = 128
    img_cols = 128

    if downsample:
        imgs_train = np.resize(imgs_train, [imgs_train.shape[0], img_rows, img_cols, imgs_train.shape[1]])
        imgs_mask_train = np.resize(imgs_mask_train,
                                    [imgs_mask_train.shape[0], img_rows, img_cols, imgs_mask_train.shape[1]])
        imgs_test = np.resize(imgs_test, [imgs_test.shape[0], img_rows, img_cols, imgs_test.shape[1]])
        imgs_mask_test_true = np.resize(imgs_mask_test_true,
                                        [imgs_mask_test_true.shape[0], img_rows, img_cols, imgs_mask_test_true.shape[1]])

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
                epochs=2,
                layers="all")


    # test
    class InferenceConfig(NodulesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = InferenceConfig()

    # TODO: 这里应该用测试集
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    Precisions = []
    Recalls = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset_val, inference_config,
                          image_id, use_mini_mask=False)
        molded_images = np.expand_dims(mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        Precisions.append(precisions)
        Recalls.append(recalls)

    print("mAP: ", np.mean(APs))
    print("mPrecision: ", np.mean(Precisions))
    print("mRecall: ", np.mean(Recalls))


config = NodulesConfig()
train, val = dataset_init()
train_test_model(train, val, config)


