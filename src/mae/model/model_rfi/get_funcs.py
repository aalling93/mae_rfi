import os
import numpy as np
import tensorflow as tf
import pywt


def get_score_subsets(
    model,
    input_images,
    decoded_images,
    image_names: list = [],
    height_kernel: int = 10,
    width_kernel: int = 50,
    save: bool = True,
    location_id: str = "",
):
    os.makedirs(f"{model.name}/subset_scores", exist_ok=True)
    if len(location_id) > 0:
        os.makedirs(f"{model.name}/subset_scores/{location_id}", exist_ok=True)

    scores_images_all = []

    for i in range(len(input_images)):
        image_temp = []
        for height in range(height_kernel, input_images.shape[1] - height_kernel, 1):
            width_temp = []
            for width in range(width_kernel, input_images.shape[2] - width_kernel, 1):
                # img_s.append(x_test_rfi[idenx][height-10:height+10,width-50:width+50,:])
                width_temp.append(
                    tf.image.ssim(
                        input_images[i][
                            height - height_kernel : height + height_kernel,
                            width - width_kernel : width + width_kernel,
                            :,
                        ],
                        decoded_images[i][
                            height - height_kernel : height + height_kernel,
                            width - width_kernel : width + width_kernel,
                            :,
                        ],
                        1,
                    ).numpy()
                )
            image_temp.append(np.array(width_temp))
            test = np.stack(image_temp, axis=0)
        if save == True:
            if len(location_id) > 0:
                np.save(
                    f"{model.name}/subset_scores/{location_id}/{image_names[i]}.npy",
                    test,
                    allow_pickle=True,
                )
            else:
                np.save(
                    f"{model.name}/subset_scores/{image_names[i]}.npy",
                    test,
                    allow_pickle=True,
                )
        scores_images_all.append(test)

    return np.array(scores_images_all)


def get_anomaly(
    score_img, lowerBound=0.85, upperBound=1.5, pad_w: int = 50, pad_l: int = 10
):
    coeffs2 = pywt.dwt2(score_img**3, "db38")
    LL, (LH, HL, HH) = coeffs2
    myMatrix = np.ma.masked_where(
        (lowerBound < score_img) & (score_img < upperBound), score_img
    )

    myMatrix = np.pad(myMatrix, ((pad_l, pad_l), (pad_w, pad_w)), mode="symmetric")

    myMatrix = np.ma.masked_where(
        (lowerBound < myMatrix) & (myMatrix < upperBound), myMatrix
    )

    return myMatrix
