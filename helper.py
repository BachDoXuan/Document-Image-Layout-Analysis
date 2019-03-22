import random
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import cv2
import tensorflow as tf

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def gen_batches_function(data_dir, image_shape, n_classes, 
                         augmentation_fn=None, debug=False):
    """
    Generate function to create batches of training data.
    Description of images and labels: must be array of type int
    images is of [height, width, 3] shape
    labels are of [height, width] shape 
        
    :param data_dir: Path to folder that contains images and labels 
        for train/val/test datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    assert os.path.isdir(data_dir), \
        data_dir + " is not a valid directory" 
    image_paths = glob(os.path.join(data_dir, 'images', '*.jpg'))
    image_filenames = os.listdir(os.path.join(data_dir, 'images'))
    if debug:
        print("data size:", len(image_paths))   
    assert len(image_paths) == len(image_filenames), \
        data_dir + " contains non-jpg files"
#    label_paths = glob(os.path.join(data_folder, 'labels', '*.jpg'))
#    assert len(image_paths) == len(label_paths), \
#        "The numbers of images and labels are not equal"
        
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(image_filenames) # shuffle images' order for each epoch

        for i in range(0, len(image_filenames), batch_size):
            images = []
            labels = []
            
            for image_filename in image_filenames[i : i + batch_size]:
                image_path = os.path.join(data_dir, 'images', image_filename)
                label_path = os.path.join(data_dir, 'labels', image_filename)
                
                if not (os.path.isfile(image_path) 
                        and os.path.isfile(image_path)
                        ):
                    raise ("image or label does not exist")
                
                # cv2: In the case of color images, the decoded images will 
                #       have the channels stored in B G R order.
                image = cv2.imread(image_path)
                label = cv2.imread(label_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                image = tf.to_float(tf.image.decode_jpeg(
#                                        tf.read_file(image_path), 
#                                        channels=3,
#                                        try_recover_truncated=True)
#                                    )
#                label = tf.to_float(tf.image.decode_jpeg(
#                                        tf.read_file(label_path), 
#                                        channels=3,
#                                        try_recover_truncated=True)
#                                    )
                
#                assert (image.shape == label.shape), \
#                        "image and label are not of the same shape"
#                if debug:
#                    print("image shape:", image.shape)
#                    print("label shape:", label.shape)
                
                # TODO: rewrite augmentation_fn
#                if augmentation_fn:
#                    image, label = augmentation_fn(image, label)
                       
                # resize images
#                if debug:
#                    print("resized image shape:", image_shape)
                    
                image = cv2.resize(image, (image_shape[1], image_shape[0]),
                                   interpolation = cv2.INTER_LINEAR)
                label = cv2.resize(label, (image_shape[1], image_shape[0]),
                                   interpolation = cv2.INTER_NEAREST)
#                image = tf.image.resize_images(image, 
#                                               tf.cast(image_shape, tf.int32), 
#                                               method=tf.image.ResizeMethod.BILINEAR)
#                label = tf.image.resize_images(label, 
#                                               tf.cast(image_shape, tf.int32), 
#                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                
                # convert label into one-hot type
                mask = np.zeros((label.shape[0], label.shape[1]), 
                                dtype = np.uint16)
                
                # TODO: write a more general code by removing [0, 0, 1] etc
                mask_1 = np.sum(label * [1,0,0], axis=2) > 0 
                mask_2 = np.sum(label * [0,1,0], axis=2) > 0
                mask_3 = np.sum(label * [0,0,1], axis=2) > 0
                mask[mask_1] = 1
                mask[mask_2] = 2
                mask[mask_3] = 3
                
                class_eye = np.eye(n_classes, dtype = np.uint8)
                label = class_eye[mask, :]
                
                # append to batch
                images.append(image)
                labels.append(label)
            
            images = np.array(images) / 127.5 - 1.0 # normalize mean of images
            labels = np.array(labels)
            if debug:
                print("image batch shape:", images.shape)
                print("label batch shape:", labels.shape)
            yield images, labels
    return get_batches_fn





