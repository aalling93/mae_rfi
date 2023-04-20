
def center_crop(img, dim):
    """
    Crop an image to the specified dimensions by taking the center portion of the image.
    
    Args:
    img (numpy.ndarray): Input image.
    dim (tuple): Tuple containing the dimensions (height, width) to which the image is to be cropped.

    Returns:
    numpy.ndarray: Cropped image.
    """
    
    # Extracting the height and width of the input image
    height,width = img.shape[0], img.shape[1]
    # Calculating crop width and height
    crop_width = dim[1] if dim[1] < img.shape[1] else img.shape[1]
    crop_height = dim[0] if dim[0] < img.shape[0] else img.shape[0]

    # Calculating the center point of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    # Calculating the half-width and half-height of the cropped image
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)

    # Cropping the image using the center point and the half-width and half-height of the cropped image
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2,:]
    
    # Returning the cropped image
    return crop_img


