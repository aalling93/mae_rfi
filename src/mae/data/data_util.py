import numpy as np


def center_crop(img, dim):
    """
    cropping img
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]
    return crop_img


def _load_data(data: str = "", crop: bool = True, imsize: tuple = (340, 500, 2)):
    data = np.load(data, allow_pickle=True)
    if crop == True:
        data = np.array(
            [
                center_crop(im, [imsize[1], imsize[0]])
                for im in data
                if (im.shape[0] >= imsize[0] and im.shape[1] >= imsize[1])
            ]
        )
    return data
