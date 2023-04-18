Gif link follows...
<img src="figs/test.png" width="750" align="center">


Kristian Aalling Sørensen
kaaso@space.dtu.dk



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



## Theory <a class="anchor" id="emi"></a>
Go back to [Table of Content](#content)


See paper (link follows.)





# Acknowledgments
 <a class="anchor" id="Acknowledgments"></a>
Myself.

The reviwer who introduced me to MAEs.


 # Licence
See License file. In short:

1. Cite us. Paper link follows.