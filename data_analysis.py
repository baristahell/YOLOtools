import os
import numpy as np
import matplotlib.pyplot as plt



def analyse_CS(source_dir):
    data = np.zeros(13)
    
    img_ids_filename = '%s/train_CS.txt' % (source_dir)
    ifs_img_ids = open(img_ids_filename)
    img_ids = ifs_img_ids.read().strip().split()
    
    for image_id in img_ids:
        image_name = os.path.splitext(os.path.basename(image_id))[0]
        anno_filename = '%s/labels/%s.txt' % (source_dir, image_name)
        ifs_anno = open(anno_filename)
        for line in ifs_anno:
            l = line.split(" ")
            data[int(l[0])] += 1

        ifs_anno.close()
    ifs_img_ids.close()
    
    return data

if __name__ == "__main__":

    # init
    source_dir = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet"
    namefile = "/home/bastienhell/darknet/cfg/cityscapes-lite-12.names"
    names = []
    num_obj_cat = []

    # get classes names
    namef = open(namefile,"r")
    for name in namef:
        names.append(name.split("\n")[0])
    namef.close()    

    # get classes count
    num_obj_cat = analyse_CS(source_dir)
    # generate bar plot
    ypos = np.arange(len(names))
    ax = plt.axes()
    ax.bar(ypos,num_obj_cat,align='center')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xticks(ypos)
    ax.set_xticklabels(names,rotation=90)
    plt.show()
