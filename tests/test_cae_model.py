import numpy as np
import unittest
from src import mae
import tensorflow as tf





class Testing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Testing, self).__init__(*args, **kwargs)
        np.random.seed(42)
        self.train_dualpol = np.random.rand(10,200,100,2)
        self.train_singpol = np.random.rand(10,200,100,1)

        self.test_dualpol = np.random.rand(10,200,100,2)
        self.test_singpol = np.random.rand(10,200,100,1)
        
       
    def test_cae_dual_pol(self):
        ae_model, __, __ = mae.model.model_rfi.model_arcitectures.modelPoolingDropout(img_size=self.train_dualpol[0].shape, latent_space_dim=25)
        self.assertEqual(ae_model.name,'AE_rfi')
    def test_cae_single_pol(self):
        ae_model, __, __ = mae.model.model_rfi.model_arcitectures.modelPoolingDropout(img_size=self.train_singpol[0].shape, latent_space_dim=25)
        self.assertEqual(ae_model.name,'AE_rfi')



if __name__ == '__main__':
    unittest.main()
