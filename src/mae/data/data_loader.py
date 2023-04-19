import numpy as np
from .data_util import center_crop


def _load_and_crop_file(
    data: str = "", crop: bool = True, imsize: tuple = (340, 500, 2)
):
    img = np.load(data, allow_pickle=True)
    if img.shape[0] >= imsize[0] and img.shape[1] >= imsize[1]:
        if crop == True:
            img = np.array([center_crop(img, [imsize[0], imsize[1]])])
        return img


def _crop_file(data, crop: bool = True, imsize: tuple = (340, 500, 2)):
    data = np.array([center_crop(data, [imsize[0], imsize[1]])])
    return data[0, :, :, :]


def _load_list_of_filenames(
    train_data: list,
    test_data: list,
    imsize: tuple = (340, 500, 2),
    only_VH: bool = False,
    test_samples: int = 100,
):
    images = np.array([np.load(im) for im in train_data])
    all_train_images = np.array(
        [
            _crop_file(im)
            for im in images
            if im.shape[0] >= imsize[0] and im.shape[1] >= imsize[1]
        ]
    )

    test_data = np.array([np.load(im) for im in test_data])
    all_test_images = np.array(
        [
            _crop_file(im)
            for im in test_data
            if im.shape[0] >= imsize[0] and im.shape[1] >= imsize[1]
        ]
    )

    np.random.shuffle(all_train_images)

    if only_VH:
        all_train_images = all_train_images[:, :, :, 1:]
        all_test_images = all_test_images[:, :, :, 1:]

    train = all_train_images[0 : int(len(all_train_images) * 0.8)]
    val = all_train_images[int(len(all_train_images) * 0.8) :]
    test = all_test_images

    del all_test_images, all_train_images

    return train, val, test


def _load_data_from_npy_path(
    train_data: str,
    test_data: str,
    imsize: tuple = (340, 500, 2),
    only_VH: bool = False,
    test_samples: int = 100,
):
    train = _load_and_crop_file(train_data, crop=True, imsize=imsize)
    if only_VH:
        train = train[:, :, :, 1:]

    val = train[int(len(train) * 0.8) :]
    train = train[0 : int(len(train) * 0.8)]

    if len(test_data) > 0:
        test = _load_and_crop_file(test_data, crop=True, imsize=imsize)
        if only_VH:
            test = test[:test_samples, :, :, 1:]

    return train, val, test


def _load_data(
    train,
    test,
    imsize: tuple = (340, 500, 2),
    only_VH: bool = False,
    test_samples: int = 100,
):

    if type(train) == str:
        train_data, val_data, test_data = _load_data_from_npy_path(
            train_data=train,
            test_data=test,
            imsize=imsize,
            only_VH=only_VH,
            test_samples=test_samples,
        )

    elif type(train) == list:
        train_data, val_data, test_data = _load_list_of_filenames(
            train_data=train,
            test_data=test,
            imsize=imsize,
            only_VH=only_VH,
            test_samples=test_samples,
        )
    else:
        print("error in load data.")

    return train_data, val_data, test_data
