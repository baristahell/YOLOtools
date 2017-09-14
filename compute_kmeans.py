# -*- coding: utf-8 -*-
import numpy as np
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sys import exit


def convert_coco_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def area(x):
    if len(x.shape) == 1:
        return x[0] * x[1]
    else:
        return x[:, 0] * x[:, 1]


def kmeans_iou(k, centroids, points, feature_size, iter_count=0, iteration_cutoff=30):

    best_clusters = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0

    npoi = points.shape[0]
    area_p = area(points)  # (npoi, 2) -> (npoi,)

    while True:
        cen2 = centroids.repeat(npoi, axis=0).reshape(k, npoi, 2)
        cdiff = points - cen2
        cidx = np.where(cdiff < 0)
        cen2[cidx] = points[cidx[1], cidx[2]]

        wh = cen2.prod(axis=2).T  # (k, npoi, 2) -> (npoi, k)
        dist = 1. - (wh / (area_p[:, np.newaxis] + area(centroids) - wh))  # -> (npoi, k)
        belongs_to_cluster = np.argmin(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters_niou = np.min(dist, axis=1)  # (npoi, k) -> (npoi,)
        clusters = [points[belongs_to_cluster == i] for i in range(k)]
        avg_iou = np.mean(1. - clusters_niou)
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_clusters = clusters
            best_avg_iou_iteration = iter_count

        #print("\nIteration {}".format(iter_count))
        #print("Average iou to closest centroid = {}".format(avg_iou))
        #print("Sum of all distances (cost) = {}".format(np.sum(clusters_niou)))

        new_centroids = np.array([np.mean(c, axis=0) for c in clusters])
        isect = np.prod(np.min(np.asarray([centroids, new_centroids]), axis=0), axis=1)
        aa1 = np.prod(centroids, axis=1)
        aa2 = np.prod(new_centroids, axis=1)
        shifts = 1 - isect / (aa1 + aa2 - isect)

        # for i, s in enumerate(shifts):
        #     print("{}: Cluster size: {}, Centroid distance shift: {}".format(i, len(clusters[i]), s))

        if sum(shifts) == 0 or iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count += 1

    # Get anchor boxes from best clusters
    anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])
    anchors = anchors[anchors[:, 0].argsort()]
    #print("k-means clustering pascal anchor points (original coordinates) \
    #\nFound at iteration {} with best average IoU: {} \
    #\n{}".format(best_avg_iou_iteration, best_avg_iou, anchors*feature_size))

    return anchors,best_avg_iou


def load_CS_dataset():
    name = 'cityscapes'
    data = []
    
    img_ids_filename = '%s/train_CS.txt' % (source_dir)
    ifs_img_ids = open(img_ids_filename)
    img_ids = ifs_img_ids.read().strip().split()
    
    for image_id in img_ids:
        image_name = os.path.splitext(os.path.basename(image_id))[0]
        anno_filename = '%s/labels/%s.txt' % (source_dir, image_name)
        ifs_anno = open(anno_filename)

        for line in ifs_anno:
            l = line.split(" ")
            l[4] = l[4].strip() # remove \n from last element
            bb = (float(l[1]),
                  float(l[2]),
                  float(l[3]),
                  float(l[4]))
            data.append(bb[2:])

        ifs_anno.close()
    ifs_img_ids.close()

    return np.array(data)

def load_VOC_dataset():
    name = 'voc'
    data = []
    
    img_ids_filename = '%s/train.txt' % (source_dir)
    ifs_img_ids = open(img_ids_filename)
    img_ids = ifs_img_ids.read().strip().split()
    
    for image_id in img_ids:
        image_name = os.path.splitext(os.path.basename(image_id))[0]
        anno_filename = '%s/labels/%s.txt' % (source_dir, image_name)
        ifs_anno = open(anno_filename)

        for line in ifs_anno:
            l = line.split(" ")
            l[4] = l[4].strip() # remove \n from last element
            bb = (float(l[1]),
                  float(l[2]),
                  float(l[3]),
                  float(l[4]))
            data.append(bb[2:])

        ifs_anno.close()
    ifs_img_ids.close()

    return np.array(data)

if __name__ == "__main__":

    # initiation
    results = []
    iou_list = []
    centroid_list = []
    k_list = range(1,11)
    
    # load CS data
    source_dir = "/mnt/bigdrive/tempdata/CityscapesData/CityScapesToDarknet"
    img_size = 1024
    CS_data = load_CS_dataset()
    feature_size=img_size/32
    
    # load VOC data
    #source_dir = "/mnt/bigdrive/tempdata/VOCdevkit/VOC2012"
    #img_size = 500
    #VOC_data = load_VOC_dataset()
    
    # run k means
    for k in k_list:
        random_data = np.random.random((1000, 2))
        centroids = np.random.random((k, 2))
        random_anchors = kmeans_iou(k, centroids, random_data, feature_size)

        # compute CS kmeans
        centroids = CS_data[np.random.choice(np.arange(len(CS_data)), k, replace=False)]
        CS_anchors_relative,best_iou = kmeans_iou(k, centroids, CS_data, feature_size)
        CS_anchors = CS_anchors_relative*13
        results.append([k,best_iou,CS_anchors])
        centroid_list.append(CS_anchors)
        iou_list.append(best_iou)

        # compute VOC kmeans
        #centroids = VOC_data[np.random.choice(np.arange(len(VOC_data)), k, replace=False)]
        #VOC_anchors_relative,best_iou = kmeans_iou(k, centroids, VOC_data, feature_size=img_size / 32)
        #VOC_anchors = VOC_anchors_relative*13
        #results.append([k,best_iou,VOC_anchors])
        #centroid_list.append(VOC_anchors)
        #iou_list.append(best_iou)

        
        print "Avg IOU at " + str(k) + " centroids: " + str(best_iou)
        
    print "K-means done"
    print "centroids : w*2,h"
    for centroid_x,centroid_y in centroid_list[9]:
        print str(centroid_x*2) + "," + str(centroid_y) + ","
        
    # plot results 
    plt.plot(k_list,iou_list)
    plt.xlabel("k")
    plt.ylabel("avg IOU")
    plt.grid()
    plt.show()
    #print centroid_list[9]
    exit(0)
