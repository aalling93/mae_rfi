
import tensorflow as tf 
from ..model.util import *
from ..model.Mae import *


def model_builder2(hp):
    """
    Build model for hyperparameters tuning
    
    hp: HyperParameters class instance
    """
    tf.keras.backend.clear_session()
    # defining a set of hyperparametrs for tuning and a range of values for each
    #filters = hp.Int(name = 'filters', min_value = 60, max_value = 230, step = 20)
    filterSize = hp.Int(name = 'filterSize', min_value = 2, max_value = 7,step=1)
    latent_space_dim = hp.Int(name = 'latentSpaceDim', min_value = 1, max_value = 20,step=1)

    
    clip = hp.Float("clipping",min_value=0.4, max_value=1, step=0.2)
    learning_rate = hp.Float("lr", min_value=1e-6, max_value=1e-2, sampling="log")

    filters = []
    for i in range(filterSize):
        filters.append(hp.Int(name = f'filters{i+1}', min_value = 60, max_value = 500, step = 50))



    drop_rate = hp.Float(name = 'dropout_prob', min_value = 0, max_value = 0.4, step = 0.05)
    laten_space_regularisation_L1 = hp.Float(name = 'laten_space_regularisation_L1', min_value = 0.00001, max_value = 0.001, step = 0.05)

    

    
    optimizer = tf.keras.optimizers.Adam(clipnorm=clip, clipvalue=clip, learning_rate=learning_rate
    )
    lr_metric = get_lr_metric(optimizer)
    



    
    return ae
