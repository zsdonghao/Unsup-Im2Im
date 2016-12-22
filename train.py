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


from model import *
from utils import *

pp = pprint.PrettyPrinter()


flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 500, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("last_saved_model", "data/Models/model_epoch_5.ckpt", "Path to the last saved model")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    """ Step 1: Train a G which generates plausible images conditioned on given class """
    z_dim = 98

    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    # z_classes = tf.placeholder(tf.float32, [FLAGS.batch_size, 2], name='z_classes')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')
    
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

    net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = 2, name ='classes_embedding')
    # z --> generator for training
    net_g, g_logits = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), FLAGS, is_train=True, reuse=False)
    # generated fake images --> discriminator
    net_d, d_logits_fake, _, d_logits_fake_class = discriminator(net_g.outputs, FLAGS, is_train=True, reuse=False)
    # real images --> discriminator
    _, d_logits_real, _, d_logits_real_class = discriminator(real_images, FLAGS, is_train=True, reuse=True)
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

    # trainable parameters for updating discriminator and generator
    # g_vars = tl.layers.get_variable_with_name('generator', True, True)
    # e_vars = tl.layers.get_variable_with_name('classes_embedding', True, True)
    # d_vars = tl.layers.get_variable_with_name('discriminator', True, True)
    t_vars = tf.trainable_variables()
    
    g_vars = [var for var in t_vars if 'generator' in var.name]
    e_vars = [var for var in t_vars if 'classes_embedding' in var.name]
    d_vars = [var for var in t_vars if 'discriminator' in var.name]

    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars + e_vars)

    """ Step 2: Train a P which is able to encode class A images to Z, and allow G to reconstruct the images """


    """ Step 3: Input images of class A, output images of class B """


    """ """
    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.998)
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    
    if os.path.exists(FLAGS.last_saved_model):
        saver.restore(sess, FLAGS.last_saved_model)

    # # load checkpoints
    # print("[*] Loading checkpoints...")
    # model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    # save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    # # load the latest checkpoints
    # net_g_name = os.path.join(save_dir, 'net_g.npz')
    # net_d_name = os.path.join(save_dir, 'net_d.npz')
    # if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
    #     print("[!] Loading checkpoints failed!")
    # else:
    #     net_g_loaded_params = tl.files.load_npz(name=net_g_name)
    #     net_d_loaded_params = tl.files.load_npz(name=net_d_name)
    #     tl.files.assign_params(sess, net_g_loaded_params, net_g)
    #     tl.files.assign_params(sess, net_d_loaded_params, net_d)
    #     print("[*] Loading checkpoints SUCCESS!")

    # TODO: use minbatch to shuffle and iterate
    class1_files, class2_files, images = load_data(FLAGS.dataset)
    all_files = class1_files + class2_files
    shuffle(all_files)
    print "all_files", len(all_files)
    total_batches = len(all_files)/FLAGS.batch_size
    print "Total_batces", total_batches
    for epoch in range(FLAGS.epoch):
        for bn in range(0, total_batches):
            batch_files = all_files[bn*FLAGS.batch_size : (bn + 1) * FLAGS.batch_size]
            batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            batch_z_classes = [0 if images[file_name]['Male'] == True else 1 for file_name in batch_files ]
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
                save_path = saver.save(sess, "data/Models/model_epoch_{}.ckpt".format(epoch))
                generated_samples = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z, 
                    z_classes : batch_z_classes,
                })[0]

                generated_samples_other_class = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z, 
                    z_classes : [0 if batch_z_classes[i] == 1 else 1 for i in range(len(batch_z_classes))],
                })[0]
                
                for i in range(0, 10):
                    
                    real_images_255 = batch_images[i]
                    scipy.misc.imsave( 'data/samples/real_{}_{}.jpg'.format(i, batch_z_classes[i]) , real_images_255) 

                    
                    fake_images_255 = generated_samples[i]
                    scipy.misc.imsave('data/samples/fake_image_{}_{}.jpg'.format(i, batch_z_classes[i]), fake_images_255)

                    fake_images_255_other_class = generated_samples_other_class[i]
                    scipy.misc.imsave('data/samples/fake_other_{}_{}.jpg'.format(i, batch_z_classes[i]), fake_images_255_other_class)

                    combined_image = [fake_images_255] + [np.zeros((64, 5, 3))] + [fake_images_255_other_class]
                    combined_image = np.concatenate( combined_image, axis = 1 )
                    scipy.misc.imsave('data/samples/combined_{}.jpg'.format(i), combined_image)

    # data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
    # print len(data_files), data_files[0:10]
    
    # TODO: shuffle sample_files each epoch
    # sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.sample_size, z_dim)).astype(np.float32)

    # iter_counter = 0
    # for epoch in range(FLAGS.epoch):
    #     #shuffle data
    #     shuffle(data_files)
    #     print("[*]Dataset shuffled!")

    #     # update sample files based on shuffled data
    #     sample_files = data_files[0:FLAGS.sample_size]
    #     sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
    #     sample_images = np.array(sample).astype(np.float32)
    #     print("[*]Sample images updated!")

    #     # load image data
    #     batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

    #     for idx in xrange(0, batch_idxs):
    #         batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
    #         # get real images
    #         batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
    #         batch_images = np.array(batch).astype(np.float32)
    #         batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
    #         start_time = time.time()
    #         # updates the discriminator
    #         errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
    #         # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
    #         for _ in range(2):
    #             errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
    #         print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
    #                 % (epoch, FLAGS.epoch, idx, batch_idxs,
    #                     time.time() - start_time, errD, errG))
    #         sys.stdout.flush()

    #         iter_counter += 1
    #         if np.mod(iter_counter, FLAGS.sample_step) == 0:
    #             # generate and visualize generated images
    #             img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
    #             '''
    #             img255 = (np.array(img) + 1) / 2 * 255
    #             tl.visualize.images2d(images=img255, second=0, saveable=True,
    #                             name='./{}/train_{:02d}_{:04d}'.format(FLAGS.sample_dir, epoch, idx), dtype=None, fig_idx=2838)
    #             '''
    #             save_images(img, [8, 8],
    #                         './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
    #             print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
    #             sys.stdout.flush()

    #         # if np.mod(iter_counter, FLAGS.save_step) == 0:
    #         #     # save current network parameters
    #         #     print("[*] Saving checkpoints...")
    #         #     img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
    #         #     model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
    #         #     save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    #         #     if not os.path.exists(save_dir):
    #         #         os.makedirs(save_dir)
    #         #     # the latest version location
    #         #     net_g_name = os.path.join(save_dir, 'net_g.npz')
    #         #     net_d_name = os.path.join(save_dir, 'net_d.npz')
    #         #     # this version is for future re-check and visualization analysis
    #         #     net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)
    #         #     net_d_iter_name = os.path.join(save_dir, 'net_d_%d.npz' % iter_counter)
    #         #     tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
    #         #     tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
    #         #     tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
    #         #     tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)
    #         #     print("[*] Saving checkpoints SUCCESS!")

def load_data(dataset):
    if dataset == 'celebA':
        attr_file = os.path.join("./data", dataset, "list_attr_celeba.txt")
        attr_rows = open(attr_file).read().split('\n')
        attr_names = attr_rows[1].split()

        images = {}
        for img_row in attr_rows[2:]:
            row = img_row.split()
            if len(row) == 0:
                break
            img_name = row[0]
            attr_flags = row[1:]
            row_dic = {}
            for i, attr_name in enumerate(attr_names):
                if attr_flags[i] == "1":
                    row_dic[attr_name] = True
                else:
                    row_dic[attr_name] = False
            images[os.path.join("./data", dataset,img_name)] = row_dic
        
        # return images
        class1_files = [ name for name in images if images[name]['Male'] == True]
        class2_files = [ name for name in images if images[name]['Male'] == False]

        shuffle(class1_files)
        shuffle(class2_files)

        min_length = min(len(class1_files), len(class2_files))
        
        class1_files = class1_files[0:min_length]
        class2_files = class2_files[0:min_length]


        return class1_files, class2_files, images


if __name__ == '__main__':
    # load_data("celebA")
    tf.app.run()
