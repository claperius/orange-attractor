#!/usr/bin/env python3

################################################################################
#
# Mikolaj Sitarz 2019
# Apache License 2.0
#
# Demonstration code for article https://orange-attractor.eu/?p=145
#
################################################################################


from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np
import argparse



def get_args():
    parser = argparse.ArgumentParser("decompose image by k-means in RGB space")
    parser.add_argument("input_file")
    parser.add_argument("--n-clusters", default=3, type=int, help="number of clusters (default 3)")
    parser.add_argument("--rnd-state", default=1, type=int, help="initial state for pseudorandom number generator (default 1)")
    parser.add_argument("--n-samples", default=1000, type=int, help="number of samples taken from the picture (default 1000)")
    parser.add_argument("--single-cluster", type=int , help="saves image with single cluster - all other clusters are removed")
    return parser.parse_args()



def read_image(fname):
    img = Image.open(fname)
    img.load()
    return np.asarray(img, dtype='uint8')



def replace_label(a, labels, label_value, value=255):
    'replace all values in array `a` with given `value` at all the the places where labels[k] == label_value '

    # mask keeps "1" on place of label_value and "0" on all other places
    mask = (labels == label_value).astype('uint8')

    # reversed mask
    rmask = mask ^ 1

    if value == 0:
        return a * rmask
    else:
        return a * rmask + value * mask


def decompose_colors(a):
    r = a[:, 0]  # red channel
    g = a[:, 1]  # green channel
    b = a[:, 2]  # blue channel

    return r, g, b



def save_cluster_boost(shape, a, labels, cluster_number):
    'save image, replace colors of all pixels belonging to the "cluster_number" with red (#FF0000)'
    
    fname = 'cluster-{:02}.jpg'.format(cluster_number)
    print('saving: {}'.format(fname))

    # replace all pixels from the cluster with red color
    
    r, g, b = decompose_colors(a)

    r = replace_label(r, labels, cluster_number, 255)
    g = replace_label(g, labels, cluster_number, 0)
    b = replace_label(b, labels, cluster_number, 0)

    # stack 3 channels back together
    out = np.stack((r, g, b), axis=-1)

    out = np.reshape(out, shape)
    img = Image.fromarray(out, 'RGB')
    img.save(fname)

    

def save_clusters_boost(shape, a, args, kmeans):
    'save images for all clusters'
    
    labels = kmeans.predict(a)
    for cluster_number in range(args.n_clusters):
        save_cluster_boost(shape, a, labels, cluster_number)
    

        
def save_single_cluster(shape, a, args, kmeans):
    fname = 'single-cluster.jpg'
    print('saving: {}'.format(fname))

    r, g, b = decompose_colors(a)

    labels = kmeans.predict(a)

    # replace all other clusters with white color
    for cluster in range(args.n_clusters):

        if cluster == args.single_cluster:
            continue
        
        r = replace_label(r, labels, cluster, 255)
        g = replace_label(g, labels, cluster, 255)
        b = replace_label(b, labels, cluster, 255)

    out = np.stack((r, g, b), axis=-1)
    out = np.reshape(out, shape)
    img = Image.fromarray(out, 'RGB')
    img.save(fname)

    
        
def fit(a, args):

    # get shuffled samples from the array
    a = shuffle(a[:args.n_samples], random_state=args.rnd_state)
    
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.rnd_state)
    kmeans.fit(a)
    return kmeans


def main():
    args = get_args()
    a_in = read_image(args.input_file)
    a_wrk = a_in.reshape((a_in.shape[0] * a_in.shape[1], 3))  # flatten array

    kmeans = fit(a_wrk, args)

    if args.single_cluster is not None:
        save_single_cluster(a_in.shape, a_wrk, args, kmeans)
    else:
        save_clusters_boost(a_in.shape, a_wrk, args, kmeans)

    
if __name__ == '__main__':
    main()
    
