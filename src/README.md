# Source code for CoraL Beaching experiments

The dependencies for the ResNet source code is defined in the ```requirements.txt``` file. 

## Dataset Class

```dataset.py```

Defines the dataset classes used for the experiments. There is a regular Coral Bleaching dataset, a dataset class loaded from a metadata dataframe, 
and a dataset loaded from metadata with corresponding SST values. 

## Training/Evaluation scripts

```main.py```

Defines the main training script used to train the regular ResNet-50 model on our dataset. The script can be run in either train or eval mode by specifying 
the args `--mode train` or `--mode eval`. Training will save a checkpoint to the `./checkpoints` folder as well as the dataset splits to `./splits` folder. 
Evaluation will load a checkpoint and dataset splits, and evaluate the model on the test set. The confusion matrix will be saved to `./images`. 

Depending on the data used, you may want to change the variables:
- To change the coralnet soure to use for training and testing: sources = ["CuraCao Coral Reef Assessment 2023 CUR"] -> train_model()
- To make the label names on the confustion matrix match the dataset: class_names = ["healthy", "bleached"] -> evaluate_model()

Also make sure the checkpoint names match when saving/loading checkpoints so the correct one is being evaluated. 

```main_sst.py```

This script works the same way as the ```main.py``` script, but uses the dataset with image, sst pairs to train the model, and trains the hybrid ResNet-50 model instead. 
The script can also be run in train or eval mode, and has generally the same inputs and outputs. 

## Hybrid ResNet-50 model

We design a hybrid CoralBleachingModel which takes in both image and numerical data as input. Uses a ResNet-50 for image feature extraction, and uses a
Simple MLP for numberical feature extraction. Both features are concatenated and passed through a final MLP to obtain model outputs. 

## DenseNet

The jupyter notebook ```denseNet.ipynb``` contains the source code for the DenseNet experiments. 
