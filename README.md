# Semantic Segmentation

This is a small project dìon semantic segmentation using a synthetic dataset obtained from Unreal Engine.
The task is to create a synthetic dataset from Unreal Engine using the plugin [Unrealcv](https://unrealcv.org).
The dataset is composed by RGBD (RGB + depth) images and ground truth. The environment is a simple room with some couches, two TVs, two plants and some tables.
Once collected the dataset it is possible to detect and localize objects in images.
This can be done using a semantic segmentation network. Semantic segmentation networks basically perform pixel-wise classification assigning for each pixel its class.

# Description of the repository
Inside semantic-seg the main files are:
- generate_dataset.py : Script for generating dataset interacting with the unrealcv server.
- write_dataset_to_tf.py : Read the dataset written previously and store in tfrecord files.
- train.py : network training using a modified version of the [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)

# Installation

Clone the repo and install requirements:
```
git clone https://github.com/EmanueleGhelfi/semantic-seg/
cd semantic-seg
conda create -n semantic-seg
pip install -r requirements.txt
```

Download the Unreal enviroment from [here](https://drive.google.com/open?id=1stYziULUXthkDaK0Bi0A6474u_hc_Lip).
Open the project with unreal, install the plugin [Unrealcv](https://unrealcv.org) following the instructions.

# Run

## Step 1: Dataset generation.

Press the Play button in the unreal editor. Configure the `generate_dataset.py` script in order to connect to the correct port.
Launch:
```
python generate_dataset.py
```
This should save the rgb images, the depth images and associated labels inside the folder `dataset`.
Only some classes are kept, see `conf.json`.

## Step 2: Tfrecord conversion

Convert the generated dataset to tfrecord files.
```
python write_dataset_to_tf.py
```
This should save the training, validation and test sets inside dataset/tfrecord/.

## Step 3: Training

Train the selected semantic segmentation network:
```
python train.py
```
