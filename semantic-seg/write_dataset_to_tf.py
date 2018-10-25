import os
from random import shuffle
import glob
import sys
import math

import scipy.misc
import tensorflow as tf
import numpy as np
import imageio
import image_helper

cwd = os.getcwd()
IMG_PATH = cwd+"/dataset/rgb/"
DEPTH_PATH = cwd+"/dataset/depth/"
LABEL_PATH = cwd+"/dataset/label/"
RGBD_PATH = cwd+"/dataset/rgbd/"
TRAIN_PATH = cwd+"/dataset/train/"
VAL_PATH = cwd+"/dataset/validation/"
TEST_PATH = cwd+"/dataset/test/"

# get file and sort by name
image_list = sorted(glob.glob(IMG_PATH+"*.png"))
depth_list = sorted(glob.glob(DEPTH_PATH+"*.npy"))
label_list = sorted(glob.glob(LABEL_PATH+"*.png"))

print(image_list[0:11], depth_list[0:11], label_list[0:11])

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

if not os.path.exists("dataset/tfrecord/train"):
  os.makedirs("dataset/tfrecord/train/")

if not os.path.exists("dataset/tfrecord/test"):
  os.makedirs("dataset/tfrecord/test/")

if not os.path.exists("dataset/tfrecord/validation"):
  os.makedirs("dataset/tfrecord/validation/")

train_filenames = ['dataset/tfrecord/train/train','dataset/tfrecord/validation/validation','dataset/tfrecord/test/test']  # address to save the TFRecords file
imgs = [train_img, val_img, test_img]
depths = [train_depths, val_depths, test_depths]
labels = [train_labels, val_labels, test_labels]
paths = [TRAIN_PATH, VAL_PATH, TEST_PATH]

NUM_IMAGE_PER_FILE = 10

for i,train_filename in enumerate(train_filenames):
    # open the TFRecords file
    depths_i = depths[i]
    imgs_i = imgs[i]
    labels_i = labels[i]
    path_i = paths[i]

    shard_idx = 0

    num_shard = math.ceil(len(imgs_i)/NUM_IMAGE_PER_FILE)

    for s in range(num_shard):

      print("Writing to " + train_filename+str(s)+".tfrecord")
      writer = tf.python_io.TFRecordWriter(train_filename+str(s)+".tfrecord")

      for j in range(NUM_IMAGE_PER_FILE):

        idx = s*NUM_IMAGE_PER_FILE+j

        if idx >= len(imgs_i):
          break

        # print how many images are saved every 1000 images
        if not j % 100:
            print('data: {}/{}'.format(idx, len(imgs_i)))
            sys.stdout.flush()
        # Load the image removing alpha channel
        img = load_image(imgs_i[idx])[:,:,:3]
        img_shape = img.shape
        
        depth = np.reshape(np.load(depths_i[idx]), (img_shape[0],img_shape[1],1))

        img_4d = np.concatenate((img, depth), axis=2)

        label = load_image(labels_i[idx])[:,:,:3]
        label = image_helper.convert_from_color_segmentation(label)

        # Create a feature
        feature = {'train/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                'train/image': _bytes_feature(tf.compat.as_bytes(img_4d.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # save image as npy and label
        #np.save(path_i+f"rgbd/image{i}.npy", img_4d)
        #imageio.imwrite(path_i+f"label/image{i}.png", label)

        
    writer.close()
    sys.stdout.flush()