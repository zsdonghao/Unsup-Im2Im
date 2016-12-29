import os
from random import shuffle

def load_data(dataset):
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

        shuffle(class1_files)
        shuffle(class2_files)

        min_length = min(len(class1_files), len(class2_files))
        
        class1_files = class1_files[0:min_length]
        class2_files = class2_files[0:min_length]


        return class1_files, class2_files, class_flag
