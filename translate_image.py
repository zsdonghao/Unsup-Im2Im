import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
import argparse
import data_loader

from model import *
from utils import *

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("z_dim", 98, "Size of Noise embedding")
flags.DEFINE_integer("class_embedding_size", 2, "Size of class embedding")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "data/Models", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

def main():
    # parser = argparse.ArgumentParser()
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')
    
    if FLAGS.class_embedding_size != None:
        net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = FLAGS.class_embedding_size, name ='classes_embedding')
    else:
        net_z_classes = InputLayer(inputs = tf.one_hot(z_classes), name ='classes_embedding')

    net_p = imageEncoder(real_images, is_train = False)
    net_g, g_logits = generator(tf.concat(1, [net_p.outputs, net_z_classes.outputs]), is_train=False)

    sess=tf.Session()
    sess.run(tf.initialize_all_variables())

    net_g_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g.npz'.format(FLAGS.dataset))
    net_e_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_e.npz'.format(FLAGS.dataset))
    net_p_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_p.npz'.format(FLAGS.dataset))

    if not (os.path.exists(net_g_name) and os.path.exists(net_e_name) and os.path.exists(net_p_name)):
        print("[!] Loading checkpoints failed!")
        return
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        net_p_loaded_params = tl.files.load_npz(name=net_p_name)
        
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)
        tl.files.assign_params(sess, net_p_loaded_params, net_p)
        
        print("[*] Loading checkpoints SUCCESS!")

    class1_files, class2_files, class_flag = data_loader.load_data(FLAGS.dataset, split = "test")
    all_files = class1_files + class2_files
    shuffle(all_files)

    batch_files = all_files[0:FLAGS.batch_size]
    batch_images = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0, random_flip = True) for batch_file in batch_files]
    batch_z_classes = [1 if class_flag[file_name] == True else 0 for file_name in batch_files ]

    gen_images = sess.run(net_g.outputs, feed_dict = {
        z_classes : batch_z_classes,
        real_images : batch_images
    })

    combine_and_save_image_sets([batch_images, gen_images], FLAGS.sample_dir)

    print "[*] Translation Complete"

if __name__ == '__main__':
    main()