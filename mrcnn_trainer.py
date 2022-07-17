from tensorflow.python.keras import backend as K
from tensorflow.python.client import device_lib
from telegram_callback import TelegramCallback
from tensorflow.keras.callbacks import *
from os.path import join
from mrcnn import utils

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn.config import Config

import imgaug.augmenters as iaa
import mrcnn.model as modellib
import tensorflow as tf
import numpy as np
import warnings
import random
import os

from functools import wraps

SEED = 37
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2,
                                  allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
K.set_session(session)
config.gpu_options.per_process_gpu_memory_fraction = 0.4

# Root directory of the project
ROOT_DIR = os.path.abspath("")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# # Download weights file
# if not os.path.exists(WEIGHTS_PATH):
#     utils.download_trained_weights(WEIGHTS_PATH)

PATH_DATASET = '/media/b/0ff41202-390d-4105-b6ed-e6aa791973aa/home/mapping-challenge/from_kaggle'

PATH_TRAIN_IMG = join(PATH_DATASET, 'images')
PATH_TRAIN_ANNOT = join(PATH_DATASET, 'annotations_train.json')


PATH_VAL_IMG = join(PATH_DATASET, 'images')
PATH_VAL_ANNOT = join(PATH_DATASET, 'annotations_val.json')


WEIGHTS_PATH = os.path.join(ROOT_DIR, "pretrained_weights.h5")
# WEIGHTS_PATH = "/home/b/Рабочий стол/Mask-RCNN, Mapping " \
#                "Challenge/logs/crowdai-mapping-challenge20220714T1155/mask_rcnn_crowdai-mapping-challenge_0022.h5 "


def trycatch(func):
    """ Обертывает декорированную функцию в try-catch. Если функция не работает, выведите исключение. """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as e:
            print(f"Exception in {func.__name__}: {e}")

    return wrapper


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return print([x.name for x in local_device_protos])


def iaa_augmentation():
    def sometimes(aug): return iaa.Sometimes(0.5, aug)

    augmenters = [
        # Horizontal flips.
        # Vertical flips.
        sometimes([iaa.Fliplr(0.5), iaa.Flipud(0.5)]),

        # sometimes(iaa.OneOf([  # weather augmentation
        #     iaa.Snowflakes(flake_size=(0.2, 0.4), speed=(0.01, 0.07)),
        #     iaa.Rain(speed=(0.3, 0.5)),
        # ])),

        # sometimes(iaa.OneOf([  # blur or sharpen
        #     iaa.GaussianBlur(sigma=(0.0, 0.3)),
        #     iaa.Sharpen(alpha=(0.0, 0.1)),
        # ])),

        # Add gaussian noise.
        # sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),

        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        #     iaa.Affine(
        #         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #         translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #         rotate=(-25, 25),
        #         shear=(-8, 8)
        #     ),
        #    iaa.OneOf([iaa.Affine(rotate=90),
        #          iaa.Affine(rotate=180),
        #          iaa.Affine(rotate=270)]),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # Strengthen or weaken the contrast in each image.

        sometimes(iaa.OneOf([  # brightness or contrast
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.ContrastNormalization((0.75, 1.5)),
        ])),
    ]

    return iaa.Sequential(augmenters, random_order=True)

    # def train_head():
    #     print("Training network heads")
    #     model.train(dataset_train, dataset_val,  # augmentation=augmentation,
    #                 learning_rate=TrainConfig.LEARNING_RATE,
    #                 epochs=40,  # custom_callbacks=[tg_callback, rlr, lc],
    #                 layers='heads')

    # print("Training all network layers")
    # model.train(dataset_train, dataset_val, augmentation=augmentation,
    #             learning_rate=TrainConfig.LEARNING_RATE, epochs=80,
    #             custom_callbacks=[tg_callback, rlr], layers='all')


class MappingChallengeDataset(utils.Dataset):
    def load_dataset(self, dataset_dir, load_train=True, return_coco=True):
        """ Загружает набор данных, выпущенный для crowdAI Mapping Challenge(https://www.crowdai.org/challenges/mapping-challenge).
            Параметры:
                - dataset_dir : корневой каталог набора данных (может указывать на папку train/val)
                - load_small : булево значение, которое сигнализирует, нужно ли загружать в память аннотации для всех изображений,
                               или только небольшое их подмножество должно быть загружено в память.

        """
        self.load_train = load_train
        if self.load_train:
            annotation_path = os.path.join(dataset_dir, "annotations_train.json")
        else:
            annotation_path = os.path.join(dataset_dir, "annotations_val.json")

        image_dir = os.path.join(dataset_dir, "images/")
        print("Annotation Path ", annotation_path)
        print("Image Dir ", image_dir)
        assert os.path.exists(annotation_path) and os.path.exists(image_dir)

        self.coco = COCO(annotation_path)
        self.image_dir = image_dir

        # Load all classes (Only Building in this version)
        classIds = self.coco.getCatIds()

        # Load all images
        image_ids = list(self.coco.imgs.keys())

        # register classes
        for _class_id in classIds:
            self.add_class("crowdai-mapping-challenge", _class_id, self.coco.loadCats(_class_id)[0]["name"])

        # Register Images
        for _img_id in image_ids:
            assert (os.path.exists(os.path.join(image_dir, self.coco.imgs[_img_id]['file_name'])))
            self.add_image(
                "crowdai-mapping-challenge", image_id=_img_id,
                path=os.path.join(image_dir, self.coco.imgs[_img_id]['file_name']),
                width=self.coco.imgs[_img_id]["width"],
                height=self.coco.imgs[_img_id]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(
                    imgIds=[_img_id],
                    catIds=classIds,
                    iscrowd=None)))

        if return_coco:
            return self.coco

    def load_mask(self, image_id):
        """ Загружает маску экземпляра для заданного изображения
              Эта функция преобразует маску из формата coco в формат
              растровое изображение [высота, ширина, экземпляр].
            Параметры:
                - image_id : идентификатор ссылки для данного изображения
            Возвращает:
                masks : Массив bool формы [высота, ширина, экземпляры] с
                    одна маска на экземпляр
                class_ids : одномерный массив classIds соответствующих масок экземпляров.
                    (В этой версии задачи он будет иметь форму [instances] и всегда будет заполнен class-id класса "Building").
        """

        image_info = self.image_info[image_id]
        assert image_info["source"] == "crowdai-mapping-challenge"

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "crowdai-mapping-challenge.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                # Ignore the notion of "is_crowd" as specified in the coco format
                # as we donot have the said annotation in the current version of the dataset

                instance_masks.append(m)
                class_ids.append(class_id)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MappingChallengeDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Возвращает ссылку на конкретное изображение
            В идеале эта функция должна возвращать URL-адрес
            но в данном случае мы просто вернем image_id
        """
        return "crowdai-mapping-challenge::{}".format(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Преобразование аннотации, которая может быть полигонами, в RLE без сжатия.
        :return: двоичная маска (двумерный массив numpy)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Преобразование аннотации, которая может быть полигонами, несжатым RLE или RLE, в двоичную маску.
        :return: двоичная маска (массив numpy 2D)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


def load_dataset():
    # Training dataset.
    dataset_train = MappingChallengeDataset()
    dataset_train.load_dataset(PATH_DATASET, load_train=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MappingChallengeDataset()
    dataset_val.load_dataset(PATH_DATASET, load_train=False)
    dataset_val.prepare()

    print("___TRAIN DATASET___: \nImages: {}\nClasses: {}".format(
        len(dataset_train.image_ids), dataset_train.class_names))
    print("\n___VAL DATASET___: \nImages: {}\nClasses: {}".format(
        len(dataset_val.image_ids), dataset_val.class_names))

    return dataset_train, dataset_val


@trycatch
def main():
    # print all available devices
    get_available_devices()

    class CustomConfig(Config):
        # Give the configuration a recognizable name
        NAME = "crowdai-mapping-challenge"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 5

        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # 1 Background + 1 Building

        STEPS_PER_EPOCH = 100
        VALIDATION_STEPS = 50

        IMAGE_MAX_DIM = 320
        IMAGE_MIN_DIM = 320

    config = CustomConfig()
    config.display()

    dataset_train, dataset_val = load_dataset()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    print("Loading weights ", WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH, by_name=True)
#     model.load_weights(WEIGHTS_PATH, by_name=True, exclude=[
#         "mrcnn_class_logits", "mrcnn_bbox_fc",
#         "mrcnn_bbox", "mrcnn_mask"])

    # create config telegram
    tg_config = {
        'token': '1641590760:AAEgQWjGo_nEpx0p1tWioBrONT0OyR8tPKM',  # bot token+
        'telegram_id': -587107216,  # telegram_id
    }
    tg_name = config.NAME + '\n' + 'BATCH=' + str(config.IMAGES_PER_GPU) + ' STEPS_PER_EPOCH=' + str(
        config.STEPS_PER_EPOCH) + ' LEARNING_RATE=' + str(config.LEARNING_RATE)

    # telegram callback
    tg_callback = TelegramCallback(tg_config, name=tg_name)
    # reduce lr callback
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1,
                            mode='min', epsilon=0.001, min_delta=1e-4, cooldown=2, min_lr=1e-6)
    # early stopping callback
    es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=6, mode='min',
                       restore_best_weights=True, verbose=1)

    augmentation = iaa_augmentation()

    # model.train(dataset_train, dataset_val, # augmentation=augmentation,
    #             learning_rate=config.LEARNING_RATE / 10, period=15,
    #             epochs=2, custom_callbacks=[tg_callback, rlr, es],
    #             layers='all')

    # Training - Stage 1
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40, period=5, custom_callbacks=[tg_callback],
    #             layers='heads')

    # # Training - Stage 2
    # # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, augmentation=augmentation,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=120, period=15, custom_callbacks=[tg_callback],
    #             layers='4+')
    #
    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tune all layers")
    model.train(dataset_train, dataset_val, augmentation=augmentation,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=60, period=2, custom_callbacks=[tg_callback, rlr],
                layers='heads')

    # session.close()


if __name__ == "__main__":
    main()
