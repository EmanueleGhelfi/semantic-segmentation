from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import glob
from imageio import imwrite

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

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



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=288, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=288, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet50", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Load the data
print("Loading the data ...")
batch_size = 5
filenames = tf.placeholder(tf.string, shape=[None])
dataset_tf = tf.data.TFRecordDataset(filenames)
# shuffle before heavy transformations
dataset_tf = dataset_tf.shuffle(buffer_size=1000)
dataset_tf = dataset_tf.map(__parse_function)  # Parse the record into tensors.
dataset_tf = dataset_tf.repeat()  # Repeat the input indefinitely.
dataset_tf = dataset_tf.batch(batch_size)
dataset_tf = dataset_tf.prefetch(3)
iterator = dataset_tf.make_initializable_iterator()
next_example, next_label = iterator.get_next()

# transform to one hot notation since it is required for computing all quantities
next_label_rev_one_hot = tf.cast(next_label, tf.uint8)
next_label = tf.one_hot(tf.cast(next_label, tf.uint8),depth=num_classes, axis=3)

print(next_label.get_shape())

train_names = glob.glob(os.getcwd()+"/dataset/tfrecord/train/*.tfrecord")
test_names = glob.glob(os.getcwd()+"/dataset/tfrecord/test/*.tfrecord")
val_names = glob.glob(os.getcwd()+"/dataset/tfrecord/validation/*.tfrecord")

# Compute your softmax cross entropy loss
net_input = next_example
net_output = next_label

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

im_argmax = tf.argmax(network, axis=3)

one_hot_img = tf.one_hot(im_argmax, depth=num_classes, axis=3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

iou, uiou = tf.metrics.mean_iou(labels=next_label_rev_one_hot, predictions=im_argmax, num_classes=num_classes)

mean_per_class_accuracy, uacc = tf.metrics.mean_per_class_accuracy(labels=next_label_rev_one_hot, predictions=im_argmax, num_classes=num_classes)

precision, uprec = tf.metrics.precision(labels=net_output, predictions=one_hot_img)

f1_score, uf1 = tf.contrib.metrics.f1_score(labels=net_output, predictions=one_hot_img)

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

update_ops = [uiou, uprec, uf1, uacc]
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_iou_per_epoch = []
avg_acc_per_epoch = []
avg_prec_per_epoch = []
avg_f1_score_per_epoch = []
avg_val_loss_per_epoch = []
avg_val_iou_per_epoch = []
avg_val_acc_per_epoch = []
avg_val_prec_per_epoch = []
avg_val_f1_score_per_epoch = []

# Which validation images do we want
val_indices = []

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []
    current_accs = []
    current_prec = []
    current_iou = []
    current_f1_score = []

    cnt=0

    num_iters = 100
    st = time.time()
    epoch_st=time.time()

    # initialize dataset iterator
    sess.run(iterator.initializer, feed_dict={filenames: train_names})

    # initialize local variables inside metrics 
    sess.run(tf.local_variables_initializer())

    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []
        
        print("Training.....")
        # Do the training
        img, label, _, current, _ = sess.run([next_example, next_label, opt, loss, update_ops])
        
        # debug
        if epoch == 0 and i==0:
            img = img[0]
            imwrite("trial.png", img[:,:,:3])
            imwrite("label.png", helpers.colour_code_segmentation(helpers.reverse_one_hot(label[0]), label_values))

        current_losses.append(current)
    
    f1_, prec_, acc_, iou_ = sess.run([f1_score, precision, mean_per_class_accuracy, iou])
    
    print(f"INFO: f1: {f1_} \n prec: {prec_} \n acc: {acc_} \n iou: {iou_}")
    string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
    utils.LOG(string_print)
    st = time.time()

    avg_loss_per_epoch.append(np.mean(current_losses))
    avg_iou_per_epoch.append(iou_)
    avg_acc_per_epoch.append(acc_)
    avg_prec_per_epoch.append(prec_)
    avg_f1_score_per_epoch.append(f1_)

    # Create directories if needed
    if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
        os.makedirs("%s/%04d"%("checkpoints",epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%("checkpoints",epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

        current_losses = []
        current_accs = []
        current_prec = []
        current_iou = []
        current_f1_score = []

        # Do the validation on a small set of validation images
        sess.run(iterator.initializer, feed_dict={filenames: val_names})

        # initialize local variables inside metrics 
        sess.run(tf.local_variables_initializer())

        val_iters = 20
        for i in range(val_iters):
            current_loss, _ = sess.run([loss, update_ops])
            current_losses.append(current_loss)
        
        f1_, prec_, acc_, iou_ = sess.run([f1_score, precision, mean_per_class_accuracy, iou])

        avg_val_acc_per_epoch.append(acc_)
        avg_val_f1_score_per_epoch.append(f1_)
        avg_val_iou_per_epoch.append(iou_)
        avg_val_prec_per_epoch.append(prec_)
        avg_val_loss_per_epoch.append(np.mean(current_losses))

        print(f"VALIDATION: f1: {f1_} \n prec: {prec_} \n acc: {acc_} \n iou: {iou_}")

        # print visualization of a batch
        ims, gts, output_images = sess.run([next_example, next_label, network])

        for j in range(ims.shape[0]):
            input_img = ims[0,:,:,:3]

            output_image = np.array(output_images[j])
            output_image = helpers.reverse_one_hot(output_image)
            gt = helpers.reverse_one_hot(gts[j])
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            file_name = f"img{j}"

            gt = helpers.colour_code_segmentation(gt, label_values)

            cv2.imwrite("%s/%04d/%s.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(input_img), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

        target.close()

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)

    if epoch % 5 == 0:

        fig1, ax1 = plt.subplots(figsize=(11, 8))

        val_x = np.arange(len(avg_val_acc_per_epoch))*args.validation_step

        ax1.plot(val_x, avg_val_acc_per_epoch)
        ax1.set_title("Average validation accuracy vs epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Avg. val. accuracy")


        plt.savefig('accuracy_vs_epochs.png')

        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(11, 8))

        ax2.plot(val_x, avg_val_loss_per_epoch)
        ax2.set_title("Average loss vs epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Current loss")

        plt.savefig('loss_vs_epochs.png')

        plt.clf()

        fig3, ax3 = plt.subplots(figsize=(11, 8))

        ax3.plot(val_x, avg_val_iou_per_epoch)
        ax3.set_title("Average val IoU vs epochs")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Current IoU")

        plt.savefig('iou_vs_epochs.png')

        plt.clf()

        fig4, ax4 = plt.subplots(figsize=(11, 8))

        ax4.plot(range(epoch+1), avg_acc_per_epoch)
        ax4.set_title("Average training accuracy vs epochs")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")

        plt.savefig('tra_acc_vs_epochs.png')

        plt.clf()

        fig5, ax5 = plt.subplots(figsize=(11, 8))

        ax5.plot(range(epoch+1), avg_iou_per_epoch)
        ax5.set_title("Average training iou vs epochs")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("IOU")

        plt.savefig('tra_iou_vs_epochs.png')

        plt.clf()

        fig6, ax6 = plt.subplots(figsize=(11, 8))

        ax6.plot(range(epoch+1), avg_loss_per_epoch)
        ax6.set_title("Average training loss vs epochs")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Current loss")

        plt.savefig('Tra_loss_vs_epochs.png')

        plt.clf()
        plt.close('all')




