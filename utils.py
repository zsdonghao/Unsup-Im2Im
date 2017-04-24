import numpy as np
import scipy.misc
import os
import random

import tensorflow as tf
from tensorlayer.prepro import *

FLAGS = tf.app.flags.FLAGS

def add_noise_fn(x, keep):
    """ Add noise on a image for inpainting experiments """
    ## random noise
    # x = (x + 1.) / 2.
    # x = drop(x, keep=keep)
    # x = x * 2. - 1.
    ## remove central reigion
    mis_size = 30 # 20
    s = int((FLAGS.image_size-mis_size)/2)
    e = s + mis_size
    x[s:e , s:e, :] = -1 # min of image is -1
    return x


def get_image_fn(path):
    """ Input a image path, return a image array """
    return scipy.misc.imread(path).astype(np.float)

def distort_fn(x):
    """ Data augmentation for a image """
    if FLAGS.dataset not in ['svhn_inpainting', 'mnist_svhn']:
        x = flip_axis(x, axis=1, is_random=True)

    if FLAGS.dataset == 'mnist_svhn': # no data augmentation
        # print(x.shape)
        # gap = random.randint(1, 5)
        # x = imresize(x, size=[FLAGS.image_size+gap, FLAGS.image_size+gap], interp='bilinear', mode=None)
        # x = imresize(x, size=[FLAGS.image_size+5, FLAGS.image_size+5], interp='bilinear', mode=None)
        x = imresize(x, size=[FLAGS.image_size, FLAGS.image_size], interp='bilinear', mode=None)
        return x/127.5 - 1.
    else:
        x = zoom(x, zoom_range=(0.95, 1.0), is_random=True)
        x = rotation(x, rg=10, is_random=True)
        x = imresize(x, size=[FLAGS.image_size+5, FLAGS.image_size+5], interp='bilinear', mode=None)
        x = crop(x, wrg=FLAGS.image_size, hrg=FLAGS.image_size, is_random=True)
        x = x/127.5 - 1. # [-1, 1]
        return x


#######################


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = scipy.misc.imresize(image, (resize_w, resize_w))
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        img = scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        img = scipy.misc.imread(path).astype(np.float)

    # grayscale images, single channel images
    if len(img.shape) == 2:
        img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
        img_new[:,:,0] = img
        img_new[:,:,1] = img
        img_new[:,:,2] = img
        img = img_new

    return img
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False, random_flip = False):
    img = transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)
    if random_flip:
        if random.random() > 0.5:
            img = np.fliplr(img)
    return img

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def combine_and_save_image_sets(image_sets, directory):
    for i in range(len(image_sets[0])):
        combined_image = []
        for set_no in range(len(image_sets)):
            combined_image.append( image_sets[set_no][i] )
            combined_image.append( np.zeros((image_sets[set_no][i].shape[0], 5, 3)) )
        combined_image = np.concatenate( combined_image, axis = 1 )

        scipy.misc.imsave( os.path.join( directory,  'combined_{}.jpg'.format(i) ), combined_image)
