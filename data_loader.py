import os
from random import shuffle
import tensorflow as tf
import tensorlayer as tl
import numpy as np
# flags = tf.app.flags
# FLAGS = flags.FLAGS

###########

# if "cifar10" in FLAGS.dataset:


###########

def load_data(dataset, split = "train", percentage=0.8):
    """Returns data file directory.

    Parameters
    ----------
    dataset : string, dataset name, "celebA", "youtube_videos", "obama_hillary"
    split : "train" or "test" for celebA
        Returns the directory of training files or testing files
    percentage : float
        If split is "train", the percentage of training data, default 0.8.

    Returns
    --------
    class1_files : list of training/testing file directory of class 1
    class2_files : list of training/testing file directory of class 2
    class_flag : dict of {'file directory', label[boolean or int]} for all files
    """
    # RETURNS class1 file parths, class 2 file paths, a dictionary indicating the class of a file name.
    if dataset == 'celebA':
        attr_file = os.path.join("./data", dataset, "list_attr_celeba.txt")
        attr_rows = open(attr_file).read().split('\n')
        attr_names = attr_rows[1].split()

        images = {}
        class_flag = {}
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

            class_flag[os.path.join("./data", dataset,img_name)] = row_dic['Male']
            images[os.path.join("./data", dataset,img_name)] = row_dic

        # return images
        class1_files = [ name for name in images if images[name]['Male'] == True]
        class2_files = [ name for name in images if images[name]['Male'] == False]
        # min_length = min(len(class1_files), len(class2_files))

        # train_length = int(percentage * min_length)
        #
        # if split == "train":
        #     class1_files = class1_files[0:train_length]
        #     class2_files = class2_files[0:train_length]
        # elif split == "test":
        #     class1_files = class1_files[train_length:min_length]
        #     class2_files = class2_files[train_length:min_length]


        if split == "train":
            class1_files = class1_files[0:int(len(class1_files)*percentage)]
            class2_files = class2_files[0:int(len(class2_files)*percentage)]
        elif split == "test":
            class1_files = class1_files[int(len(class1_files)*percentage):]
            class2_files = class2_files[int(len(class2_files)*percentage):]


        shuffle(class1_files)
        shuffle(class2_files)

        return class1_files, class2_files, class_flag

    # if dataset == 'celebA_inpainting': #TODO
    #     class1_data_dir = './data/celebA'
    #     # class1_files = []
    #     # for dirpath, dirnames, filenames in os.walk(class1_data_dir):
    #     #     for filename in [f for f in filenames if f.endswith(".jpg")]:
    #     #         class1_files.append(os.path.join(dirpath, filename))
    #     #
    #     #     class_flag[os.path.join("./data", dataset,img_name)] = row_dic['Male']
    #     #     images[os.path.join("./data", dataset,img_name)] = row_dic
    #     class1_files = []
    #     for dirpath, dirnames, filenames in os.walk(class1_data_dir):
    #         for filename in [f for f in filenames if f.endswith(".jpg")]:
    #             class1_files.append(os.path.join(dirpath, filename))
    #
    #     class2_files = list(class1_files)
    #
    #     # min_length = min(len(class1_files), len(class2_files))
    #     #
    #     # train_length = int(percentage * len(class2_files))
    #     #
    #     # if split == "train":
    #     #     class1_files = class1_files[0:train_length]
    #     #     class2_files = class2_files[0:train_length]
    #     # elif split == "test":
    #     #     class1_files = class1_files[train_length:min_length]
    #     #     class2_files = class2_files[train_length:min_length]
    #
    #     if split == "train":
    #         class1_files = class1_files[0:int(len(class1_files)*percentage)]
    #         class2_files = class2_files[0:int(len(class2_files)*percentage)]
    #     elif split == "test":
    #         class1_files = class1_files[int(len(class1_files)*percentage):]
    #         class2_files = class2_files[int(len(class2_files)*percentage):]
    #
    #     shuffle(class1_files)
    #     shuffle(class2_files)
    #
    #     class_flag = None
    #
    #     return class1_files, class2_files, class_flag


    # if dataset == 'youtube_videos':
    #     data_dir = './data/faces/imgs/aligned_images_DB'
    #
    #     subd1 = "Gabi_Zimmer"
    #     subd2 = "Natasha_McElhone"
    #
    #     class1_files = []
    #     for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir, subd1)):
    #         for filename in [f for f in filenames if f.endswith(".jpg")]:
    #             class1_files.append(os.path.join(dirpath, filename))
    #
    #     class2_files = []
    #     for dirpath, dirnames, filenames in os.walk(os.path.join(data_dir, subd2)):
    #         for filename in [f for f in filenames if f.endswith(".jpg")]:
    #             class2_files.append(os.path.join(dirpath, filename))
    #
    #     shuffle(class1_files)
    #     shuffle(class2_files)
    #
    #     min_length = min(len(class1_files), len(class2_files))
    #
    #     class1_files = class1_files[0:min_length]
    #     class2_files = class2_files[0:min_length]
    #
    #
    #     if split == "train":
    #         class1_files = class1_files[0:train_length]
    #         class2_files = class2_files[0:train_length]
    #     elif split == "test":
    #         class1_files = class1_files[train_length:min_length]
    #         class2_files = class2_files[train_length:min_length]
    #
    #     class_flag = {}
    #     for file_name in class1_files:
    #         class_flag[file_name] = True
    #
    #     for file_name in class2_files:
    #         class_flag[file_name] = False
    #
    #     return class1_files, class2_files, class_flag
    #     # print class2_files

    if dataset == "obama_hillary":
        class1_data_dir = './data/obama_hillary/obama'
        class1_files = []
        for dirpath, dirnames, filenames in os.walk(class1_data_dir):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                class1_files.append(os.path.join(dirpath, filename))

        class2_data_dir = './data/obama_hillary/hillary'
        class2_files = []
        for dirpath, dirnames, filenames in os.walk(class2_data_dir):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                class2_files.append(os.path.join(dirpath, filename))


        if split == "train":
            class1_files = class1_files[0:int(len(class1_files)*percentage)]
            class2_files = class2_files[0:int(len(class2_files)*percentage)]
        elif split == "test":
            class1_files = class1_files[int(len(class1_files)*percentage):]
            class2_files = class2_files[int(len(class2_files)*percentage):]

        shuffle(class1_files)
        shuffle(class2_files)

        class_flag = {}
        for file_name in class1_files:
            class_flag[file_name] = True

        for file_name in class2_files:
            class_flag[file_name] = False

        return class1_files, class2_files, class_flag

    # if dataset == "cifar10_inpainting":
    #     ## download dataste
    #     os.system('mkdir data/cifar10_inpainting')
    #     import scipy.misc
    #     X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(
    #                                         shape=(-1, 32, 32, 3), plotable=False)
    #     print('Saving image to data/cifar10_inpainting')
    #     for i in range(len(X_train)):
    #         scipy.misc.imsave('data/{}/train_{}.jpg'.format(dataset, i), X_train[i])
    #     for i in range(len(X_test)):
    #         scipy.misc.imsave('data/{}/test_{}.jpg'.format(dataset, i), X_test[i])
    #
    #     ##
    #     data_dir = './data/cifar10_inpainting/'
    #     file_list = tl.files.load_file_list(path=data_dir, regx='\.(jpg)', printable=False)
    #
    #     class1_files = []
    #     for f in file_list:
    #         if split == 'train' and 'train' in f:
    #             class1_files.append("data/cifar10_inpainting/" + f)
    #         if split == 'test' and 'test' in f:
    #             class1_files.append("data/cifar10_inpainting/" + f)
    #
    #     class2_files = list(class1_files)
    #
    #     shuffle(class1_files)
    #     shuffle(class2_files)
    #
    #     class_flag = None
    #
    #     return class1_files, class2_files, class_flag

    if dataset == "svhn_inpainting":

        data_dir = './data/svhn/'
        file_list = tl.files.load_file_list(path=data_dir, regx='\.(jpg)', printable=False)

        class1_files = []
        for f in file_list:
            # if split == 'train' and 'train' in f:
            #     class1_files.append("data/svhn/" + f)
            # if split == 'test' and 'test' in f:
            class1_files.append("data/svhn/" + f)

        class2_files = list(class1_files)

        if split == "train":
            class1_files = class1_files[0:int(len(class1_files)*percentage)]
            class2_files = class2_files[0:int(len(class2_files)*percentage)]
        elif split == "test":
            class1_files = class1_files[int(len(class1_files)*percentage):]
            class2_files = class2_files[int(len(class2_files)*percentage):]

        shuffle(class1_files)
        shuffle(class2_files)

        class_flag = None

        return class1_files, class2_files, class_flag

    if dataset == "mnist_svhn":
        import scipy.misc
        # data_dir1 = './data/svhn/'
        # file_list = tl.files.load_file_list(path=data_dir1, regx='\.(jpg)', printable=False)
        # file_list = []
        # class1_files = []
        # for f in file_list:
        #     class1_files.append("data/svhn/" + f)
        #     print(f)
        data_dir = './data/svhn/'
        file_list = tl.files.load_file_list(path=data_dir, regx='\.(jpg)', printable=False)

        class1_files = []
        for f in file_list:
            # if split == 'train' and 'train' in f:
            #     class1_files.append("data/svhn/" + f)
            # if split == 'test' and 'test' in f:
            class1_files.append("data/svhn/" + f)

        if not os.path.exists('data/mnist'):
            X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
            print('Saving image to data/mnist')
            for i in range(len(X_train)):
                img = tl.prepro.imresize(X_train[i], size=[64, 64], interp='bilinear', mode=None)
                img = np.repeat(img, 3, axis=2)
                # exit(img.shape)
                scipy.misc.imsave('data/mnist/train_{}.png'.format( i), img)
            for i in range(len(X_test)):
                img = tl.prepro.imresize(X_test[i], size=[64, 64], interp='bilinear', mode=None)
                img = np.repeat(img, 3, axis=2)
                scipy.misc.imsave('data/mnist/test_{}.png'.format( i), img)

        file_list = tl.files.load_file_list(path='data/mnist', regx='\.(png)', printable=False)

        class2_files = []
        for f in file_list:
            if split == 'train' and 'train' in f:
                class2_files.append("data/mnist/" + f)
            if split == 'test' and 'test' in f:
                class2_files.append("data/mnist/" + f)
        # class2_files = []
        # for dirpath, dirnames, filenames in os.walk("./data/mnist_png/"):
        #     for filename in [f for f in filenames if f.endswith(".png")]:
        #         class2_files.append(os.path.join(dirpath, filename))

        if split == "train":
            class1_files = class1_files[0:int(len(class1_files)*percentage)]
            class2_files = class2_files[0:int(len(class2_files)*percentage)]
        elif split == "test":
            class1_files = class1_files[int(len(class1_files)*percentage):]
            class2_files = class2_files[int(len(class2_files)*percentage):]

        shuffle(class1_files)
        shuffle(class2_files)

        class_flag = {}
        for file_name in class1_files:
            class_flag[file_name] = True

        for file_name in class2_files:
            class_flag[file_name] = False

        return class1_files, class2_files, class_flag




if __name__ == '__main__':
    # load_data('youtube_videos')
    # class1_files, class2_files, class_flag = load_data(dataset="celebA", split = "test")
    class1_files, class2_files, class_flag = load_data(dataset="mnist_svhn", split = "train")
    print(class2_files[0])
    print(len(class1_files))
    print(len(class2_files))
    print(len(class_flag))
