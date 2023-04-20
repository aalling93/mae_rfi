Gif link follows...
<img src="figs/test.png" width="750" align="center">


Kristian Aalling Sørensen
kaaso@space.dtu.dk
DTU Security


# Brief Description
<a class="anchor" id="intro"></a>

This is a small github module that I made to compare a convolutional autoencoder with a masked autoencoder.
They are implemented for comparison. Neither of these two models are optimised using, e.g., hyperparamters optimization such as keras tuner. 





# Table Of Contents
<a class="anchor" id="content"></a>

-  [Introduction](#Introduction)
-  [Requirements](#Requirements)
-  [Install and Run](#Install-and-Run)
-  [Training a model  ](#use)
-  [Some theory  ](#theory)
-  [Acknowledgments](#Acknowledgments)



# Requirements
 <a class="anchor" id="Requirements"></a>

- See requirements file..

# Install and Run
 <a class="anchor" id="Install-and-Run"></a>

Currently, only git is supported... So, clone the dir.

```python
git clone git@github.com:aalling93/mae_rfi.git
```

Now go to the folder and use it as is..

## Training a model <a class="anchor" id="use"></a>
Go back to [Table of Content](#content)

The models can be training using either the **train.py** scripts, or using a notebook. In this module, the mae is trained using the **train.py** script whereas the cae is trained using a notebook. This is a simle version of the modules/models used in my paper. It would be relativly simple to also add hyperparameter optimization etc to this module. It has not been done here. 


1. Train mae
------------

```python
python3 train_no_cv.py
```


2. train cae
------------
See the **rfi_ae.ipynb** in the examples..



I am using a clearml callback that gives me the results as is during training (i.e. i am evaluating them after each try). 

This module has only been made to compare models and i have therefore not implekemtne a inferrence module in the src.



------------



## Theory <a class="anchor" id="theory"></a>
Go back to [Table of Content](#theory)

**autoencoders** 

An autoencoder is a neural network architecture that learns to encode and decode input data. It consists of an encoder that maps the input data to a compressed representation, and a decoder that maps the compressed representation back to the original input data. The goal of an autoencoder is to minimize the reconstruction error between the input data and the reconstructed output, which encourages the network to learn a compact and informative representation of the input data.

**masked autoencoders** 
Masked autoencoders differ from traditional autoencoders in that they incorporate a binary mask into the architecture. This mask indicates which features or dimensions of the input data should be masked during the encoding process. The masked dimensions are effectively ignored by the network and do not contribute to the encoded output. 

Recently, the vision transformer architecture has gained significant attention in the field of computer vision. The vision transformer consists of a series of transformer blocks, which are also used in the encoder and decoder of a masked autoencoder. In a transformer block, the input data is first split into patches, which are then linearly projected into a higher dimensional space. The resulting feature vectors are then passed through a series of multi-head attention and fully connected layers, which transform the input data into a more informative representation.

These multi-head attention layers and fully connected  layers can be seen in the Decoder and Encoder modules in the src.mae.model.modules. 


The encoder of a masked autoencoder can be seen as a series of transformer blocks that map the input data to a compressed representation, while the decoder maps the compressed representation back to the original input data.

In the context of anomaly detection, masked autoencoders can be used to detect anomalies in data that contain missing or corrupted values. The masked dimensions represent the missing or corrupted values, and the network is trained to reconstruct the input data based on the available information while ignoring the masked dimensions. If the network encounters a test data point that deviates significantly from the training data distribution, it may struggle to reconstruct the input data accurately, resulting in a high reconstruction error. This high error indicates that the test data point is anomalous or abnormal.

**cae vs mae** 

Both masked autoencoders with transformer blocks and convolutional autoencoders are popular architectures for image reconstruction tasks. In the case of images that are 340x500x2 in size, both models can be used effectively, but there are some differences between the two architectures that are worth noting.

First, let's compare the general characteristics of the two models. A convolutional autoencoder is a type of neural network that uses convolutional layers in the encoder and decoder to process the input image. The convolutional layers are designed to learn features that are invariant to translation, rotation, and scaling, which is well-suited for image processing tasks. On the other hand, a masked autoencoder with transformer blocks uses transformer blocks in the encoder and decoder to transform the input image into a compressed representation.  In terms of model size, a convolutional autoencoder is typically smaller than a masked autoencoder with transformer blocks. This is because the transformer blocks in the latter model are more computationally expensive than the convolutional layers in the former model.  In general, both models can perform well for image reconstruction tasks on images that are 340x500x2 in size. However, the choice between the two models may depend on the specific characteristics of the data and the task at hand. For example, if the images contain complex spatial dependencies, a masked autoencoder with transformer blocks may perform better than a convolutional autoencoder. On the other hand, if the images have more local features and less complex spatial dependencies, a convolutional autoencoder may be more appropriate. While convolutional autoencoders are generally smaller and more computationally efficient, masked autoencoders with transformer blocks can learn more complex representations that may be better suited for images with complex spatial dependencies. 


**transformer block**

The transformer block is a key building block in the transformer architecture, which has been shown to be effective for a wide range of natural language processing (NLP) tasks. In recent years, the transformer architecture has also been applied to image processing tasks, with promising results.

In the context of image processing, the transformer block operates on 2D grids of image features, and consists of two main components: multi-head self-attention and feedforward processing. The multi-head self-attention component allows the model to learn to attend to different parts of the image features based on their relevance to each other, while the feedforward component captures non-linear interactions between different features.

One key advantage of using transformer blocks for image processing tasks is that they can capture long-range dependencies between different parts of the image. In contrast, convolutional neural networks (CNNs) typically rely on local filters to capture local patterns in the image, and may struggle to capture long-range dependencies. This can be especially important for tasks such as object detection and segmentation, where the location of objects in the image may depend on long-range relationships between different parts of the image. 

However, Radio Frequency Interference patters are locally connected. I.e., the anomalies are connected spatially. The advantages of the transformer is therfore somewhat non-existing for RFI detection.

Ane potential disadvantage of using transformer blocks for image processing tasks is that they can be computationally expensive and require a large number of parameters. This is because each position in the feature map is connected to every other position via the self-attention mechanism, leading to a quadratic increase in the number of parameters with respect to the input size. In contrast, CNNs typically have a fixed number of parameters that is determined by the size of the filters and the number of channels.

To mitigate this issue, several approaches have been proposed to reduce the computational cost of the self-attention mechanism in the transformer architecture, such as sparse attention, where only a subset of the positions attend to each other, or axial attention, where the attention is computed along only one dimension at a time. Additionally, hybrid architectures that combine transformer blocks with convolutional layers have also been proposed, allowing the model to capture both local and global patterns in the image.


See paper (link follows.)





# Acknowledgments
 <a class="anchor" id="Acknowledgments"></a>
Myself.

The reviwer who introduced me to MAEs.


 # Licence
See License file. In short:

1. Cite us. Paper link follows.