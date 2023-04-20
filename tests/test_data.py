import numpy as np
import unittest
from src import mae
from src.mae.data import Data
import tensorflow as tf
import glob
from pathlib import Path




class Testing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Testing, self).__init__(*args, **kwargs)
        test_image_folder = "test_data"
        self.files = glob.glob(f"{Path(__file__).parent}/{test_image_folder}/*.npy")
        self.img_size1 = (340,500,2)
        self.img_size2 = (240,200,1)
    
    def test_list_of_files_1(self):
        data = Data()
        data.load_data(
            train_data=self.files,
            test_data=self.files,
            imsize=self.img_size1,
            only_VH=False,
            test_samples = 1
        )
        self.assertEqual(data.train.shape,(1, 340, 500, 2))
        self.assertEqual(data.test.shape,(2, 340, 500, 2))
        self.assertEqual(data.val.shape,(1, 340, 500, 2))

    def test_list_of_files_2(self):
        data = Data()
        data.load_data(
            train_data=self.files,
            test_data=self.files,
            imsize=self.img_size2,
            only_VH=True,
            test_samples = 1
        )
        
        self.assertEqual(data.train.shape,(1, 240, 200, 1))
        self.assertEqual(data.test.shape,(2, 240, 200, 1))
        self.assertEqual(data.val.shape,(1, 240, 200, 1))

    


if __name__ == '__main__':
    unittest.main()
