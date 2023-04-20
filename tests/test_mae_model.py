import numpy as np
import unittest
from src import mae
import tensorflow as tf
from src.mae.CONSTANTS import *




class TestEncoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEncoder, self).__init__(*args, **kwargs)

    def test_encoder_dual_pol(self):
        encoder = mae.model.modules.create_encoder(
            num_heads=ENC_NUM_HEADS,
            num_layers=ENC_LAYERS,
            enc_transformer_units=ENC_TRANSFORMER_UNITS,
            epsilon=LAYER_NORM_EPS,
            enc_projection_dim=ENC_PROJECTION_DIM,
            dropout=DROPOUT_RATE,
        )
        self.assertEqual(encoder.name,'mae_encoder')




class TestDecoder(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDecoder, self).__init__(*args, **kwargs)
        np.random.seed(42)
        self.train_dualpol = np.random.rand(10,200,100,2)
        self.train_singpol = np.random.rand(10,200,100,1)

    def test_decoder_dual_pol(self):
        decoder = mae.model.modules.create_decoder(
                        num_layers=DEC_LAYERS,
                        num_heads=DEC_NUM_HEADS,
                        image_size=self.train_dualpol[0].shape,
                        dropout=DROPOUT_RATE,
                        num_patches=NUM_PATCHES,
                        enc_projection_dim=ENC_PROJECTION_DIM,
                        dec_projection_dim=DEC_PROJECTION_DIM,
                        epsilon=LAYER_NORM_EPS,
                        dec_transformer_units=DEC_TRANSFORMER_UNITS,
                    )
        self.assertEqual(decoder.name,'mae_decoder')

    def test_decoder_single_pol(self):
        decoder = mae.model.modules.create_decoder(
                        num_layers=DEC_LAYERS,
                        num_heads=DEC_NUM_HEADS,
                        image_size=self.train_singpol[0].shape,
                        dropout=DROPOUT_RATE,
                        num_patches=NUM_PATCHES,
                        enc_projection_dim=ENC_PROJECTION_DIM,
                        dec_projection_dim=DEC_PROJECTION_DIM,
                        epsilon=LAYER_NORM_EPS,
                        dec_transformer_units=DEC_TRANSFORMER_UNITS,
                    )
        self.assertEqual(decoder.name,'mae_decoder')
        


if __name__ == '__main__':
    unittest.main()
