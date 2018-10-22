import os
from random import shuffle
import glob
import sys

import scipy.misc
import tensorflow as tf
import numpy as np
import imageio

cwd = os.getcwd()
IMG_PATH = cwd+"/dataset/rgb/"
DEPTH_PATH = cwd+"/dataset/depth/"
LABEL_PATH = cwd+"/dataset/label/"
RGBD_PATH = cwd+"/dataset/rgbd/"
TRAIN_PATH = cwd+"/dataset/train/"
VAL_PATH = cwd+"/dataset/validation/"
TEST_PATH = cwd+"/dataset/test/"

image_list = glob.glob(IMG_PATH+"*.png")
depth_list = glob.glob(DEPTH_PATH+"*.npy")
label_list = glob.glob(LABEL_PATH+"*.png")

print(image_list, depth_list, label_list)

shuffle_data = True  # shuffle the addresses before saving
    
# Divide the hata into 60% train, 20% validation, and 20% test
train_img = image_list[0:int(0.6*len(image_list))]
train_labels = label_list[0:int(0.6*len(label_list))]
train_depths = depth_list[0:int(0.6*len(depth_list))]
val_img = image_list[int(0.6*len(image_list)):int(0.8*len(image_list))]
val_depths = depth_list[int(0.6*len(depth_list)):int(0.8*len(depth_list))]
val_labels = label_list[int(0.6*len(label_list)):int(0.8*len(label_list))]
test_img = image_list[int(0.8*len(image_list)):]
test_labels = label_list[int(0.8*len(label_list)):]
test_depths = depth_list[int(0.8*len(depth_list)):]

def load_image(file):
    import imageio
    return imageio.imread(file) 

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filenames = ['train.tfrecords','validation.tfrecords','test.tfrecords']  # address to save the TFRecords file
imgs = [train_img, val_img, test_img]
depths = [train_depths, val_depths, test_depths]
labels = [train_labels, val_labels, test_labels]
paths = [TRAIN_PATH, VAL_PATH, TEST_PATH]

for i,train_filename in enumerate(train_filenames):
    # open the TFRecords file
    depths_i = depths[i]
    imgs_i = imgs[i]
    labels_i = labels[i]
    path_i = paths[i]

    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(imgs_i)):
        # print how many images are saved every 1000 images
        if not i % 100:
            print('data: {}/{}'.format(i, len(imgs_i)))
            sys.stdout.flush()
        # Load the image removing alpha channel
        img = load_image(imgs_i[i])[:,:,:3]
        
        depth = np.reshape(np.load(depths_i[i]), (480,640,1))

        img_4d = np.concatenate((img, depth), axis=2)

        label = load_image(labels_i[i])[:,:,:3]
        # Create a feature
        feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                'train/image': _bytes_feature(tf.compat.as_bytes(img_4d.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # save image as npy and label
        np.save(path_i+f"rgbd/image{i}.npy", img_4d)
        imageio.imwrite(path_i+f"label/image{i}.png", label)

        
    writer.close()
    sys.stdout.flush()