# RDL-unet

#Introduction
Design enhanced filtering convolution, random convolution
Depthwise separable convolution is used to train the model on a food image dataset.

#Software Architecture
Based on PyTorch

#Installation tutorial

1. Download the latest version of PyTorch

#Instructions for use

1. Dataset original image storage address: data/JPEGImages mask storage address: data/SegmentationClass
2. Run train.exe directly, where the train_image folder stores the effect images during the training process
3. Save weights in the params folder
4. The test_all. py and randomly select 20 groups for testing,
The test results are stored in the random_test_image folder

#Instructions for use

1. Dataset original image storage address: data/JPEGImages mask storage address: data/SegmentationClass
2. Run train.exe directly, where the train_image folder stores the effect images during the training process
3. Save weights in the params folder
4. Test test_all. py and randomly select 20 groups for testing,
The test results are stored in the random_test_image folder


#Explanation of evaluation indicators

1. Train. py automatically generates the training loss as a loss_data. csv file during training
2. Train. py automatically saves the training images during the training process. It is necessary to convert image data into numerical data through the metric.py file and then visualize the graph.
3. Generate multiple training data through metrics to compare the performance of accuracy and precision metrics.