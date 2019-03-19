import re
import random
import numpy as np
import os
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
#from keras.utils import get_file
from tqdm import tqdm
from sklearn.model_selection import train_test_split
#from IPython import embed
import cv2


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [
        vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',  # noqa
                os.path.join(
                    vgg_path,
                    vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def maybe_download_mobilenet_weights(alpha_text='1_0', rows=224):
    base_weight_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'  # noqa
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
    weigh_path = base_weight_path + model_name
    weight_path = get_file(model_name,
                           weigh_path,
                           cache_subdir='models')
    return weight_path

# TODO: rewrite gen_batches_function to generate batches of images and labels
# to train and evaluate
def gen_batches_function(data_dir, image_shape, n_classes, 
                         augmentation_fn=None):
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
                assert (image.shape == label.shape), \
                        "image and label are not of the same shape"
                print("image shape:", image.shape)
                print("label shape:", label.shape)
                
                # TODO: rewrite augmentation_fn
                if augmentation_fn:
                    image, label = augmentation_fn(image, label)
                
        
                # resize images
                image = cv2.resize(image, (image_shape[1], image_shape[0]),
                                   interpolation = cv2.INTER_LINEAR)
                label = cv2.resize(label, (image_shape[1], image_shape[0]),
                                   interpolation = cv2.INTER_NEAREST)
                
                # convert label into one-hot type
                mask = np.zeros((label.shape[0], label.shape[1]), 
                                dtype = np.uint16)
                
                # TODO: write a more general code by removing [0, 0, 1] etc
                mask_1 = np.sum(label * [0,0,1], axis=2) > 0 
                mask_2 = np.sum(label * [0,1,0], axis=2) > 0
                mask_3 = np.sum(label * [1,0,0], axis=2) > 0
                mask[mask_1] = 1
                mask[mask_2] = 2
                mask[mask_3] = 3
                
                class_eye = np.eye(n_classes, dtype = np.uint8)
                label = class_eye[mask, :]
                
                # append to batch
                images.append(image)
                labels.append(label)

            yield np.array(images) / 127.5 - 1.0, np.array(labels)

    return get_batches_fn


def gen_test_output(
        sess,
        logits,
        image_pl,
        data_folder,
        learning_phase,
        image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in sorted(
            glob(os.path.join(data_folder, 'image_2', '*.png')))[:]:
        image = scipy.misc.imresize(
            scipy.misc.imread(image_file, mode='RGB'), image_shape)
        pimg = image / 127.5 - 1.0
        im_softmax = sess.run(
            tf.nn.softmax(logits),
            {image_pl: [pimg],
             learning_phase: 0})
        im_softmax = im_softmax[:, 1].reshape(
            image_shape[0], image_shape[1])
        segmentation = (
            im_softmax > 0.5).reshape(
            image_shape[0],
            image_shape[1],
            1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(
        runs_dir,
        data_dir,
        sess,
        image_shape,
        logits,
        learning_phase,
        input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, input_image, os.path.join(
            data_dir, 'data_road/testing'), learning_phase, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
