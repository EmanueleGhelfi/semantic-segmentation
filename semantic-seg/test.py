import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from imageio import imwrite
from utils import utils, helpers
import glob
from builders import model_builder

# helper function for reading from iterator
def __parse_function(item):
     
    features = {"train/image": tf.FixedLenFeature([], tf.string, default_value=""),
              "train/label": tf.FixedLenFeature([], tf.string, default_value="")}
    parsed_features = tf.parse_single_example(item, features)
    img = tf.decode_raw(parsed_features['train/image'], tf.float32)
    img = tf.reshape(img,(480,640,4))
    label = tf.decode_raw(parsed_features["train/label"], tf.uint8)
    label = tf.cast(tf.reshape(label, (480,640,1)), tf.float32)

    # random crop
    img_label = tf.concat((img, label), axis=2)
    img_label_crop = tf.random_crop(img_label, [288,288,5])
    img = img_label_crop[:,:,:4]
    label = img_label_crop[:,:,4]
    label = tf.cast(tf.reshape(label, (288, 288)), tf.uint8)
    return img, label

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=288, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=288, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Load the data
print("Loading the data ...")
batch_size = 5
filenames = tf.placeholder(tf.string, shape=[None])
dataset_tf = tf.data.TFRecordDataset(filenames)
# shuffle before heavy transformations
dataset_tf = dataset_tf.shuffle(buffer_size=1000)
dataset_tf = dataset_tf.map(__parse_function)  # Parse the record into tensors.
dataset_tf = dataset_tf.batch(batch_size)
dataset_tf = dataset_tf.prefetch(3)
iterator = dataset_tf.make_initializable_iterator()
next_example, next_label = iterator.get_next()

test_names = glob.glob(os.getcwd()+"/dataset/tfrecord/test/*.tfrecord")

# transform to one hot notation since it is required for computing all quantities
next_label_rev_one_hot = tf.cast(next_label, tf.uint8)
next_label = tf.one_hot(tf.cast(next_label, tf.uint8),depth=num_classes, axis=3)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = next_example
net_output = next_label

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

im_argmax = tf.argmax(network, axis=3)

one_hot_img = tf.one_hot(im_argmax, depth=num_classes, axis=3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

iou, uiou = tf.metrics.mean_iou(labels=next_label_rev_one_hot, predictions=im_argmax, num_classes=num_classes)

mean_per_class_accuracy, uacc = tf.metrics.mean_per_class_accuracy(labels=next_label_rev_one_hot, predictions=im_argmax, num_classes=num_classes)

precision, uprec = tf.metrics.precision(labels=net_output, predictions=one_hot_img)

f1_score, uf1 = tf.contrib.metrics.f1_score(labels=net_output, predictions=one_hot_img)

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

update_ops = [uiou, uprec, uf1, uacc]

sess.run(tf.global_variables_initializer())

sess.run(tf.local_variables_initializer())

# initialize dataset iterator
sess.run(iterator.initializer, feed_dict={filenames: test_names})

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Create directories if needed
if not os.path.isdir("%s"%("Test")):
        os.makedirs("%s"%("Test"))

target=open("%s/test_scores.csv"%("Test"),'w')
target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
current_losses = []

# Run testing on ALL test images
i = 0
while True:
        try:
                img, label, _, current, _ = sess.run([next_example, next_label, opt, loss, update_ops])
        except Exception:
                break

        # debug
        if i==0:
                img = img[0]
                imwrite("trial.png", img[:,:,:3])
                imwrite("label.png", helpers.colour_code_segmentation(helpers.reverse_one_hot(label[0]), label_values))

        current_losses.append(current)
    
f1_, prec_, acc_, iou_ = sess.run([f1_score, precision, mean_per_class_accuracy, iou])
    
print(f"TEST: f1: {f1_} \n prec: {prec_} \n acc: {acc_} \n iou: {iou_} \n loss: {np.mean(current_losses)}")
