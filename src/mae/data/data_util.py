
def center_crop(img, dim):
    """
    cropping img
    """
    #print(img.shape)
    height,width = img.shape[0], img.shape[1]
    # process crop width and height for max available dimension
    crop_width = dim[1] if dim[1] < img.shape[1] else img.shape[1]
    crop_height = dim[0] if dim[0] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2,:]
    #print(crop_img.shape)
    return crop_img


