# GAS14-Attention-CNN

## How to train
1. Select desired model from **checkpoints** folder. 
2. Each model contains its generator. Copy it into **models/model.py** from **checkpoints/YOUR_MODEL/model.py**.
3. python run_model.py 
        **--phase** train
        **--checkpoints** checkpoints/YOUR_MODEL
        **--datalist** PATH_TO_TRAIN_DATALIST
        **--val_datalist** PATH_TO_VALIDATION_DATALIST
        **--restore_ckpt** INITIAL_CHECKPOINT_NUMBER_FOR_TRANSFER_LEARNING
4. Script will save learning process into **checkpoints/YOUR_MODEL**. Code will save latest checkpoint and ***best*** checkpoint in terms of PSNR metric.

## How to test
1. Select desired model from **checkpoints** folder. 
2. Each model contains its generator. Copy it into **models/model.py** from **checkpoints/YOUR_MODEL/model.py**.
3. python run_model.py 
        **--phase** test
        **--checkpoints** checkpoints/YOUR_MODEL
        **--test_datalist** PATH_TO_TEST_DATALIST
        **--output_path** PATH_TO_SAVE_PROCESSED_TEST_IMAGES
4. Script will automatically restore ***best*** checkpoint and test it. Code will save test statistics into **checkpoints/YOUR_MODEL**.

## Requirements
* python 3.7.3

* absl-py==0.9.0
* astor==0.8.1
* cycler==0.10.0
* decorator==4.4.2
* gast==0.3.3
* google-pasta==0.2.0
* grpcio==1.27.2
* h5py==2.10.0
* imageio==2.8.0
* Keras-Applications==1.0.8
* Keras-Preprocessing==1.1.0
* kiwisolver==1.1.0
* Markdown==3.2.1
* matplotlib==3.2.1
* networkx==2.4
* numpy==1.18.2
* opencv-python==4.2.0.32
* Pillow==7.0.0
* progressbar==2.5
* progressbar2==3.50.1
* protobuf==3.11.3
* pyparsing==2.4.6
* python-dateutil==2.8.1
* python-utils==2.4.0
* PyWavelets==1.1.1
* scikit-image==0.16.2
* scipy==1.4.1
* six==1.14.0
* tensorboard==1.14.0
* tensorflow-estimator==1.14.0
* tensorflow-gpu==1.14.0
* termcolor==1.1.0
* Werkzeug==1.0.0
* wrapt==1.12.1