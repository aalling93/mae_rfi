
<img src="figs/test.png" width="750" align="center">


Kristian Aalling Sørensen
kaaso@space.dtu.dk



# Brief Description
<a class="anchor" id="intro"></a>

Her skal vi skrive en fancy beskrivelse


# Table Of Contents
<a class="anchor" id="content"></a>

-  [Introduction](#Introduction)
-  [Requirements](#Requirements)
-  [Install and Run](#Install-and-Run)
-  [Examples ](#use)
-  [EMI ](#emi)
-  [Acknowledgments](#Acknowledgments)



# Requirements
 <a class="anchor" id="Requirements"></a>

- See requirements filde..
# Install and Run
 <a class="anchor" id="Install-and-Run"></a>

Currently, only git is supported. Later pypi will be aded. So, clone the dir.




## Make reconstructions <a class="anchor" id="use"></a>
Go back to [Table of Content](#content)




1. Load module
------------

```python

```


1. Load data
------------

```python


```

2. Make reconstructions
----------------
```python

```


3.Do PCA
```python

```




4. comapre results
```python

```



------------



## Theory <a class="anchor" id="emi"></a>
Go back to [Table of Content](#content)


#### Masked Autoencoders:

The masked autoencoders (MAE), probaly inspired bu Googles Bert, is simplae. Remove portion of the data and predict removed contect. Masked autoencoders are a more general type of denoising AE. In convolutional based AEs, it is not straightforward to make masked tokens or positional embeddings since it is on a regular grid pr. def. However, in ViT this is not a problem. 

I.e., the Vit based MAE does NOT use convolutions, even on images.

During encoder, it only sees the visible parts. Then, adter the encoder, the masked patched are introduced and the decoder sees both the masks and the unmased patches. Masking can be applied to both pre-training and fine-tuning, improving accuracy and reducing training computation. 

Note that the encoder of MAE is only used to encode the visual patches. The encoded patches are then concatenated with mask tokens, which the decoder (which also consists of Transformer blocks) takes as input. Each mask token is a shared, learned vector that indicates the presence of a missing patch to be predicted. Fixed sin/cos position embeddings are added both to the input of the encoder and the decoder.

However, Loss function (MSE) is calculated only on invisible tokens.

**Pretraining**
Pretrainnig is when you mask a lot of data! (like 75%)

**Infeerence**
The mae is used on the uncorrupted images, e.g., no patches. 

After pre-training, one “throws away” the decoder used to reconstruct pixels, and one uses the encoder for fine-tuning/linear probing. 


**It is simple:**

mask random patches of the input image and reconstruct the missing pixels. Using an AE.

The enocder only see the visible subsets (the ones that are not masked). The lightweight(!!) decoder that reconstructs the original image from the latent space.

**Encoder**
The encoder only sees the visible patches. If you use 75% masking, you only see and train on 25% data. This reduced training time (and memory consumption) greatly! This is very good for scaling.  Masking, e.g., 75 % of the original image gives good resutls. The models therefore scales well. 


**Decoder**
The decoder reconstructs pixels.  The original paper found that the decoder structure is very important and should not be a simple MLP as in BERT. 


learn stronger representations with local self-attention in the decoder.







# Acknowledgments
 <a class="anchor" id="Acknowledgments"></a>
Us

 # Licence
See License file. In short:

1. Cite us