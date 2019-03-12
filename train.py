import os
import pprint
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from random import shuffle
import argparse

pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
# flags.DEFINE_float("weight_decay", 1e-5, "Weight decay for l2 loss")
# flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use (will be center cropped) [64]")
flags.DEFINE_integer("z_dim", 100, "Size of Noise embedding")
flags.DEFINE_integer("class_embedding_size", 5, "Size of class embedding")
# flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 100, "The interval of saveing checkpoints. [200]")
flags.DEFINE_integer("imageEncoder_steps", 30000, "Number of train steps for image encoder")
flags.DEFINE_string("dataset", "svhn", "The name of dataset [celebA, obama_hillary, svhn, svhn_inpainting]")
flags.DEFINE_string("checkpoint_dir", "data/Models", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "data/samples", "Directory name to save the image samples [samples]")
# flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS
# print(FLAGS)
# print(type(FLAGS), FLAGS.save_step)
# exit()
os.system('mkdir data')
os.system('mkdir {}'.format(FLAGS.sample_dir))
os.system('mkdir {}'.format(FLAGS.sample_dir+'/step1'))
os.system('mkdir {}'.format(FLAGS.sample_dir+'/step2'))

import data_loader
# from model import *
import model
from utils import *

if FLAGS.image_size == 64:
    generator = model.generator
    discriminator = model.discriminator
    imageEncoder = model.imageEncoder
# elif FLAGS.image_size == 256:
#     generator = model.generator_256
#     discriminator = model.discriminator_256
#     imageEncoder = model.imageEncoder_256
else:
    raise Exception("image_size should be 64 or 256")

def train_ac_gan():
    z_dim = FLAGS.z_dim
    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim], name='real_images')

    # z embedding
    if FLAGS.class_embedding_size != None:
        net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = FLAGS.class_embedding_size, name ='classes_embedding')
    else:
        net_z_classes = InputLayer(inputs = tf.one_hot(z_classes, 2), name ='classes_embedding')

    # z --> generator for training
    net_g, _ = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), is_train=True, reuse=False)
    # net_g.print_params(False)
    # exit()

    # generated fake images --> discriminator
    net_d, d_logits_fake, _, d_logits_fake_class, _ = discriminator(net_g.outputs, is_train=True, reuse=False)

    # real images --> discriminator
    _, d_logits_real, _, d_logits_real_class, _ = discriminator(real_images, is_train=True, reuse=True)

    # sample_z --> generator for evaluation, set is_train to False
    net_g2, _ = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), is_train=False, reuse=True)

    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_real, tf.ones_like(d_logits_real)))    # real == 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(d_logits_real, tf.ones_like(d_logits_real), name='dreal')
    # discriminator: images from generator (fake) are labelled as 0
    # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.zeros_like(d_logits_fake)))     # fake == 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits_fake, tf.zeros_like(d_logits_fake), name='dfake')
    d_loss_class = tl.cost.cross_entropy(d_logits_real_class, z_classes, name='ce')                                           # cross-entropy
    d_loss = d_loss_real + d_loss_fake + d_loss_class

    # generator: try to make the the fake images look real (1)
    g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logits_fake, tf.ones_like(d_logits_fake)))
    g_loss_class = tl.cost.cross_entropy(d_logits_fake_class, z_classes, name='g')
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
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.5)
    sess.run(tf.initialize_all_variables())

    net_g_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g.npz'.format(FLAGS.dataset))
    net_d_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d.npz'.format(FLAGS.dataset))
    net_e_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_e.npz'.format(FLAGS.dataset))

    if not FLAGS.retrain:
        if not (os.path.exists(net_g_name) and os.path.exists(net_d_name) and os.path.exists(net_e_name)):
            print("[!] Could not load weights from npz files")
        else:
            net_g_loaded_params = tl.files.load_npz(name=net_g_name)
            net_d_loaded_params = tl.files.load_npz(name=net_d_name)
            net_e_loaded_params = tl.files.load_npz(name=net_e_name)
            tl.files.assign_params(sess, net_g_loaded_params, net_g)
            tl.files.assign_params(sess, net_d_loaded_params, net_d)
            tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)
            print("[*] Loading checkpoints SUCCESS!")
    else:
        print("[*] Retraining AC GAN")

    class1_files, class2_files, class_flag = data_loader.load_data(FLAGS.dataset, split = "train")
    # exit(class1_files[0:10])
    is_class_balance = True # class balancing

    if is_class_balance:
        total_batches = 2 * max(len(class1_files), len(class2_files)) /FLAGS.batch_size
        class1_files = np.asarray(class1_files)
        class2_files = np.asarray(class2_files)
    else:
        all_files = class1_files + class2_files
        total_batches = len(all_files)/FLAGS.batch_size
        shuffle(all_files)
        print("all_files", len(all_files))

    print("Total_batches", total_batches)

    for epoch in range(FLAGS.epoch):
        for bn in range(0, int(total_batches)):

            if is_class_balance:
                # if bn % 2 == 0:
                #     idex = get_random_int(min=0, max=len(class1_files), number=FLAGS.batch_size)
                #     batch_files = class1_files[idex]
                # else:
                #     idex = get_random_int(min=0, max=len(class2_files), number=FLAGS.batch_size)
                #     batch_files = class2_files[idex]

                idex = get_random_int(min=0, max=len(class1_files)-1, number=int(FLAGS.batch_size /2)) #/2)
                batch_files = class1_files[idex]
                idex = get_random_int(min=0, max=len(class2_files)-1, number=int(FLAGS.batch_size /2)) #/2)
                batch_files = np.concatenate((batch_files, class2_files[idex]))
            else:
                batch_files = all_files[bn*FLAGS.batch_size : (bn + 1) * FLAGS.batch_size]

            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
                # batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)

            # batch_images = [imread(batch_file, is_grayscale=0) for batch_file in batch_files]
            batch_images = threading_data(batch_files, fn=get_image_fn)

            # if FLAGS.dataset != 'mnist_svhn':
            batch_images = threading_data(batch_images, fn=distort_fn)


            if "svhn" in FLAGS.dataset: # for celebA_inpainting
                batch_images[:int(FLAGS.batch_size/2)] = threading_data(batch_images[:int(FLAGS.batch_size/2)], fn=add_noise_fn, keep=0.8)
                batch_z_classes = [0]*int(FLAGS.batch_size/2) + [1]*int(FLAGS.batch_size/2)
            else:
                batch_z_classes = [0 if class_flag[file_name] == True else 1 for file_name in batch_files ]
                # batch_images = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.image_size, is_grayscale = 0, random_flip = True) for batch_file in batch_files]

            # save_images(batch_images, [8,8], '_batch_images.png')
            # print(batch_images.shape, batch_images.min(), batch_images.max())
            # print(batch_z_classes)

            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                z_noise: batch_z,
                z_classes : batch_z_classes,
                real_images: batch_images
            })

            for _ in range(2):
                errG, _ = sess.run([g_loss, g_optim], feed_dict={
                    z_noise: batch_z,
                    z_classes : batch_z_classes,
                })

            print("d_loss={}\t g_loss={}\t epoch={}\t batch_no={}\t total_batches={}".format(errD, errG, epoch, bn, total_batches))

            if bn % FLAGS.save_step == 0:
                print("[*] Saving Models...")

                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                tl.files.save_npz(net_z_classes.all_params, name=net_e_name, sess=sess)

                # Saving after each iteration
                tl.files.save_npz(net_g.all_params, name=net_g_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name + "_" + str(epoch), sess=sess)
                tl.files.save_npz(net_z_classes.all_params, name=net_e_name + "_" + str(epoch), sess=sess)

                print("[*] Models saved")

                generated_samples = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z,
                    z_classes : batch_z_classes,
                })[0]

                generated_samples_other_class = sess.run([net_g2.outputs], feed_dict={
                    z_noise: batch_z,
                    z_classes : [0 if batch_z_classes[i] == 1 else 1 for i in range(len(batch_z_classes))],
                })[0]

                combine_and_save_image_sets( [batch_images, generated_samples, generated_samples_other_class], FLAGS.sample_dir+'/step1')

def train_imageEncoder():
    z_dim = FLAGS.z_dim

    z_noise = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    z_classes = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, ], name='z_classes')

        # real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim], name='real_images')

    if FLAGS.class_embedding_size != None:
        net_z_classes = EmbeddingInputlayer(inputs = z_classes, vocabulary_size = 2, embedding_size = FLAGS.class_embedding_size, name ='classes_embedding')
    else:
        net_z_classes = InputLayer(inputs = tf.one_hot(z_classes, 2), name ='classes_embedding')


    net_g, _ = generator(tf.concat(1, [z_noise, net_z_classes.outputs]), is_train=False, reuse=False)
    # net_g.print_params(False)
    # exit()

    net_p = imageEncoder(net_g.outputs, is_train=True)

    net_g2, _ = generator(tf.concat(1, [net_p.outputs, net_z_classes.outputs]), is_train=False, reuse=True)

    t_vars = tf.trainable_variables()
    p_vars = [var for var in t_vars if 'imageEncoder' in var.name]

        # net_d, d_logits_fake, _, d_logits_fake_class, df_gen = discriminator(net_g2.outputs, is_train = False, reuse = False)
        # _, _, _, _, df_real = discriminator(real_images, is_train = False, reuse = True)

        # p_loss_l2 = tf.reduce_mean(tf.square(tf.sub(real_images, net_g2.outputs )))
        # p_loss_df = tf.reduce_mean(tf.square(tf.sub(df_gen.outputs, df_real.outputs )))

        # p_reg_loss = None
        # for p_var in p_vars:
        #     if p_reg_loss == None:
        #         p_reg_loss = FLAGS.weight_decay * tf.nn.l2_loss(p_var)
        #     else:
        #         p_reg_loss += FLAGS.weight_decay * tf.nn.l2_loss(p_var)

        # p_loss = p_loss_l2 + (p_loss_df * 0.5) #+ p_reg_loss

    p_loss = tf.reduce_mean( tf.square( tf.sub( net_p.outputs, z_noise) ))

    p_optim = tf.train.AdamOptimizer(FLAGS.learning_rate/2, beta1=FLAGS.beta1) \
                  .minimize(p_loss, var_list=p_vars)

    sess = tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.5)

    # sess.run(tf.initialize_all_variables())
    tl.layers.initialize_global_variables(sess)

    # RESTORE THE TRAINED AC_GAN

    net_g_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g.npz'.format(FLAGS.dataset))
    # net_d_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d.npz'.format(FLAGS.dataset))
    net_e_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_e.npz'.format(FLAGS.dataset))

    if not (os.path.exists(net_g_name) and os.path.exists(net_e_name)):
        print("[!] Loading checkpoints failed!")
        return
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        # net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        net_e_loaded_params = tl.files.load_npz(name=net_e_name)

        tl.files.assign_params(sess, net_g_loaded_params, net_g2)
        # tl.files.assign_params(sess, net_d_loaded_params, net_d)
        tl.files.assign_params(sess, net_e_loaded_params, net_z_classes)

        print("[*] Loading checkpoints SUCCESS!")

    net_p_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_p.npz'.format(FLAGS.dataset))
    if not FLAGS.retrain:
        net_p_loaded_params = tl.files.load_npz(name=net_p_name)
        tl.files.assign_params(sess, net_p_loaded_params, net_p)
        print("[*] Loaded Pretrained Image Encoder!")
    else:
        print("[*] Retraining ImageEncoder")


    model_no = 0
    for step in range(0, FLAGS.imageEncoder_steps):
        batch_z_classes = [0 if random.random() > 0.5 else 1 for i in range(FLAGS.batch_size)]
        batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.sample_size, z_dim)).astype(np.float32)
            # batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
        # batch_images = sess.run(net_g.outputs, feed_dict ={z_noise: batch_z, z_classes : batch_z_classes })

        batch_images, gen_images, _, errP = sess.run([net_g.outputs, net_g2.outputs, p_optim, p_loss], feed_dict={
            z_noise : batch_z,
            z_classes : batch_z_classes,
        })

        print("p_loss={}\t step_no={}\t total_steps={}".format(errP, step, FLAGS.imageEncoder_steps))

        if step % FLAGS.sample_step == 0:
            print("[*] Sampling images")
            combine_and_save_image_sets([batch_images, gen_images], FLAGS.sample_dir+'/step2')

        if step % 2000 == 0:
            model_no += 1

        if step % FLAGS.save_step == 0:
            print("[*] Saving Model")
            tl.files.save_npz(net_p.all_params, name=net_p_name, sess=sess)
            tl.files.save_npz(net_p.all_params, name=net_p_name + "_" + str(model_no), sess=sess)
            print("[*] Model p(encoder) saved")

def main():#_):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_step', type=str, default="ac_gan",
                       help='Step of the training : ac_gan, imageEncoder')

    parser.add_argument('--retrain', type=int, default=0,
                       help='Set 0 for using pre-trained model, 1 for retraining the model')

    args = parser.parse_args()

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    FLAGS.retrain = args.retrain == 1

    if args.train_step == "ac_gan":
        train_ac_gan()

    elif args.train_step == "imageEncoder":
        train_imageEncoder()

if __name__ == '__main__':
    # load_data("celebA")
    # tf.app.run()
    main()
