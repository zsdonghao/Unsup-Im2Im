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
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("weight_decay", 1e-5, "Weight decay for l2 loss")

flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("z_dim", 98, "Size of Noise embedding")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 100, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "data/Models", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data/samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("last_saved_ac_gan", "data/Models/model_epoch_3.ckpt", "Path to the last saved model")

FLAGS = flags.FLAGS

def train_ac_gan():
    z_dim = FLAGS.z_dim
    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')
    
    # z embedding
    net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = 2, name ='classes_embedding')
    
    # z --> generator for training
    net_g, g_logits = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), FLAGS, is_train=True, reuse=False)
    
    # generated fake images --> discriminator
    net_d, d_logits_fake, _, d_logits_fake_class, _ = discriminator(net_g.outputs, FLAGS, is_train=True, reuse=False)
    
    # real images --> discriminator
    _, d_logits_real, _, d_logits_real_class, _ = discriminator(real_images, FLAGS, is_train=True, reuse=True)
    
    # sample_z --> generator for evaluation, set is_train to False
    net_g2, g2_logits = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), FLAGS, is_train=False, reuse=True)

    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_real, tf.ones_like(d_logits_real)))    # real == 1
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.zeros_like(d_logits_fake)))     # fake == 0
    d_loss_class = tl.cost.cross_entropy(d_logits_real_class, z_classes)                                                   # cross-entropy
    d_loss = d_loss_real + d_loss_fake + d_loss_class
    
    # generator: try to make the the fake images look real (1)
    g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.ones_like(d_logits_fake)))
    g_loss_class = tl.cost.cross_entropy(d_logits_fake_class, z_classes)
    g_loss = g_loss_fake + g_loss_class
    
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'generator' in var.name]
    e_vars = [var for var in t_vars if 'classes_embedding' in var.name]
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    
    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars + e_vars)
    
    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
    
    saver = tf.train.Saver()
    
    sess.run(tf.initialize_all_variables())
    
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    print "[*] Model Loaded"
    
    net_g_name = os.path.join(FLAGS.checkpoint_dir, 'net_g.npz')
    net_d_name = os.path.join(FLAGS.checkpoint_dir, 'net_d.npz')
    net_e_name = os.path.join(FLAGS.checkpoint_dir, 'net_e.npz')

    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Could not load weights from npz files")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)
        print("[*] Loading checkpoints SUCCESS!")

    class1_files, class2_files, class_flag = data_loader.load_data(FLAGS.dataset)

    all_files = class1_files + class2_files
    shuffle(all_files)
    print "all_files", len(all_files)
    total_batches = len(all_files)/FLAGS.batch_size
    print "Total_batches", total_batches
    
    for epoch in range(FLAGS.epoch):
        for bn in range(0, total_batches):
            batch_files = all_files[bn*FLAGS.batch_size : (bn + 1) * FLAGS.batch_size]
            batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)

            # Only for celebA dataset.. change this for others..
            batch_z_classes = [0 if class_flag[file_name] == True else 1 for file_name in batch_files ]
            batch_images = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]

            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                z_noise: batch_z, 
                z_classes : batch_z_classes,
                real_images: batch_images 
            })

            for i in range(0,2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={
                    z_noise: batch_z, 
                    z_classes : batch_z_classes,
                })

            print "d_loss={}\t g_loss={}\t epoch={}\t batch_no={}\t total_batches={}".format(errD, errG, epoch, bn, total_batches)

            if bn % FLAGS.save_step == 0:
                print "[*]Saving Model, sampling images..."
                save_path = saver.save(sess, "data/Models/model_ac_gan_{}.ckpt".format(epoch))
                
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                tl.files.save_npz(net_z_classes.all_params, name=net_e_name, sess=sess)

                # Saving after each iteration
                tl.files.save_npz(net_g.all_params, name=net_g_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_z_classes.all_params, name=net_e_name + "_" + str(epoch), sess=sess)

                print "[*]Weights saved"
                
                generated_samples = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z, 
                    z_classes : batch_z_classes,
                })[0]

                generated_samples_other_class = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z, 
                    z_classes : [0 if batch_z_classes[i] == 1 else 1 for i in range(len(batch_z_classes))],
                })[0]
                
                combine_and_save_image_sets( [batch_images, generated_samples, generated_samples_other_class], FLAGS.sample_dir)

def train_imageEncoder():
    z_dim = FLAGS.z_dim

    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

    net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = 2, name ='classes_embedding')
    
    net_p = imageEncoder(real_images, FLAGS)
    net_g3, g3_logits = generator(tf.concat(1, [net_p.outputs, net_z_classes.outputs]), FLAGS, is_train=False)

    t_vars = tf.trainable_variables()
    p_vars = [var for var in t_vars if 'imageEncoder' in var.name]
    
    net_d, d_logits_fake, _, d_logits_fake_class, df_gen = discriminator(net_g3.outputs, FLAGS, is_train = False, reuse = False)
    _, _, _, _, df_real = discriminator(real_images, FLAGS, is_train = False, reuse = True)

    # g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.ones_like(d_logits_fake)))
    # g_loss_class = tl.cost.cross_entropy(d_logits_fake_class, z_classes)

    p_loss_l2 = tf.reduce_mean(tf.square(tf.sub(real_images, net_g3.outputs )))
    p_loss_df = tf.reduce_mean(tf.square(tf.sub(df_gen.outputs, df_real.outputs )))
    
    p_reg_loss = None
    for p_var in p_vars:
        if p_reg_loss == None:
            p_reg_loss = FLAGS.weight_decay * tf.nn.l2_loss(p_var)
        else:
            p_reg_loss += FLAGS.weight_decay * tf.nn.l2_loss(p_var)

    p_loss = p_loss_l2 + (p_loss_df * 0.5) + p_reg_loss
    
    p_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                  .minimize(p_loss, var_list=p_vars)

    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    
    # RESTORE THE TRAINED AC_GAN
    
    net_g_name = os.path.join(FLAGS.checkpoint_dir, 'net_g.npz')
    net_d_name = os.path.join(FLAGS.checkpoint_dir, 'net_d.npz')
    net_e_name = os.path.join(FLAGS.checkpoint_dir, 'net_e.npz')

    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading checkpoints failed!")
        return
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g3)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)
        print("[*] Loading checkpoints SUCCESS!")


    net_p_name = os.path.join(FLAGS.checkpoint_dir, 'net_p.npz')

    class1_files, class2_files, class_flag = data_loader.load_data(FLAGS.dataset)
    all_files = class1_files + class2_files
    shuffle(all_files)
    print "all_files", len(all_files)
    total_batches = len(all_files)/FLAGS.batch_size
    print "Total_batces", total_batches
    for epoch in range(FLAGS.epoch):
        for bn in range(0, total_batches):
            batch_files = all_files[bn*FLAGS.batch_size : (bn + 1) * FLAGS.batch_size]
            batch_z_classes = [0 if class_flag[file_name] == True else 1 for file_name in batch_files ]
            batch_images = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]

            errP, _, gen_images = sess.run([p_loss, p_optim, net_g3.outputs], feed_dict={
                z_classes : batch_z_classes,
                real_images: batch_images
            })

            print "p_loss={}\t epoch={}\t batch_no={}\t total_batches={}".format(errP, epoch, bn, total_batches) 

            if bn % FLAGS.sample_step == 0:
                print "[*]Sampling images"
                combine_and_save_image_sets([batch_images, gen_images], FLAGS.sample_dir)

            if bn%FLAGS.save_step == 0:
                print "[*]Saving Model"
                save_path = saver.save(sess, "data/Models/model_step_2_epoch_{}.ckpt".format(epoch))    
                tl.files.save_npz(net_p.all_params, name=net_p_name, sess=sess)
                tl.files.save_npz(net_p.all_params, name=net_p_name + "_" + str(epoch), sess=sess)
                print "[*]Model p saved"

def main(_):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_step', type=str, default="ac_gan",
                       help='Step of the training : ac_gan, imageEncoder')

    parser.add_argument('--resume_from_model', type=str, default="data/Models/model_epoch_3.ckpt",
                       help='Resume Training from')

    args = parser.parse_args()


    # pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    
    FLAGS.last_saved_ac_gan = args.resume_from_model
    
    if args.train_step == "ac_gan":
        train_ac_gan()
    
    elif args.train_step == "imageEncoder":
        train_imageEncoder()

    """ Step 1: Train a G which generates plausible images conditioned on given class """

if __name__ == '__main__':
    # load_data("celebA")
    tf.app.run()
