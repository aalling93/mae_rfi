import tensorflow as tf


def ssim_loss(y_true, y_pred):
    # https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
