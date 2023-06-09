import os

import matplotlib.pyplot as plt
import tensorflow as tf
from clearml import Dataset, Logger, Task


def clearml_log_scalar(ssim, epoch, series: str = "", title: str = "SSIM"):
    """Logs scalar value to ClearML using the logger API.

    Args:
        ssim (float): Structural similarity index of the image.
        epoch (int): Current epoch number.
        series (str, optional): Series name to log under. Defaults to "".
        title (str, optional): Title of the scalar plot. Defaults to "SSIM".
    """
    
    Logger.current_logger().report_scalar(title, series, iteration=epoch, value=ssim)




def clearml_log_figure(array, epoch, series: str = "", title: str = ""):
    """Logs scalar value to ClearML using the logger API.

    Args:
        array (NxM) or (NxMx3): image 
        epoch (int): Current epoch number.
        series (str, optional): Series name to log under. Defaults to "".
        title (str, optional): Title of the scalar plot. Defaults to "".
    """
    fig = plt.figure(figsize=(10,10))
    im = plt.imshow(array,cmap='gray')
    cb = plt.colorbar(im, shrink=0.8, orientation="vertical")
    cb.set_label("Intensity")
    Task.current_task().get_logger().report_matplotlib_figure(
        title=title,
        series=series, 
        iteration=epoch,
        figure=fig
    )



def clearml_plot_model(model):
    """Plots the Keras model and logs the plot to ClearML using the logger API.

    Args:
        model (tf.keras.Model): Keras model to plot.
    """

    tf.keras.utils.plot_model(model, show_shapes=True, to_file="temp_del.png")
    fig = plt.figure(figsize=(15,60))
    plt.imshow(plt.imread('temp_del.png'))
    plt.title(f'{model.name}', fontsize=12)
    
    Task.current_task().get_logger().report_matplotlib_figure(
        title=f"{model.name}",
        series="", 
        figure=fig
    )
    os.remove("temp_del.png")

def clearml_plot_examples(original, masked, reconstruct, epoch, img_ix):
    """Plots the original, masked and reconstructed images and logs the plot to ClearML using the logger API.

    Args:
        original (np.ndarray): Original image.
        masked (np.ndarray): Masked image.
        reconstruct (np.ndarray): Reconstructed image.
        epoch (int): Current epoch number.
        img_ix (int): Index of the current image.
    """
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    im = plt.imshow(original[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 2)
    im = plt.imshow(masked[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Masked VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 3)
    im = plt.imshow(reconstruct[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 4)
    im = plt.imshow(original[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original VV: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 5)
    im = plt.imshow(masked[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Masked VV: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 6)
    im = plt.imshow(reconstruct[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed VV: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    Task.current_task().get_logger().report_matplotlib_figure(
        title=f"Img {img_ix} Debug Samples",
        series="",
        figure=fig,
        report_image=True,
        iteration=epoch,
    )



def clearml_plot_one_polari_all(original, masked,latent,reconstruct, epoch, img_ix):
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    im = plt.imshow(original, vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 2)
    im = plt.imshow(masked, vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Masked VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 3)
    im = plt.imshow(reconstruct, vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 5)
    im = plt.imshow(latent, cmap="gray")
    plt.title(f"{img_ix} latent space: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    

    Task.current_task().get_logger().report_matplotlib_figure(
        title=f"Img {img_ix} Debug Samples",
        series="",
        figure=fig,
        report_image=True,
        iteration=epoch,
    )







def clearml_plot_org_latent_recon(original, masked, reconstruct, epoch, img_ix):
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    im = plt.imshow(original[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 2)
    im = plt.imshow(masked[:, :], cmap="gray")
    plt.title(f"{img_ix} latent space : {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 3)
    im = plt.imshow(reconstruct[:, :, 1], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed VH: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 4)
    im = plt.imshow(original[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original VV: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 6)
    im = plt.imshow(reconstruct[:, :, 0], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed VV: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    Task.current_task().get_logger().report_matplotlib_figure(
        title=f"Img {img_ix} Debug Samples",
        series="",
        figure=fig,
        report_image=True,
        iteration=epoch,
    )



def clearml_plot_org_latent_recon_single_pil(original, latent, reconstruct, epoch, img_ix):
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    im = plt.imshow(original[:, :], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Original: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 2)
    im = plt.imshow(latent[:, :], cmap="gray")
    plt.title(f"{img_ix} latent space : {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 3)
    im = plt.imshow(reconstruct[:, :], vmin=0, vmax=1, cmap="gray")
    plt.title(f"{img_ix} Reconstructed: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 4)
    im = plt.imshow(abs(original-reconstruct), cmap="gray")
    plt.title(f"{img_ix} Difference: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    plt.subplot(2, 3, 6)
    im = plt.imshow(abs(original-reconstruct)-abs(original-reconstruct).min(), cmap="gray")
    plt.title(f"{img_ix} Difference 2: {epoch:03d}")
    cb = plt.colorbar(im, shrink=0.9, orientation="vertical")
    cb.set_label("Intensity")

    Task.current_task().get_logger().report_matplotlib_figure(
        title=f"Img {img_ix} Debug Samples",
        series="",
        figure=fig,
        report_image=True,
        iteration=epoch,
    )



def clearml_plot_graph(
    values,
    title: str = "",
    series: str = "",
    xlabel: str = "Step",
    ylabel: str = "Learning rate",
):
    fig = plt.figure()
    plt.plot(values)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    Task.current_task().get_logger().report_matplotlib_figure(
        title=title, series=series, figure=fig
    )




def clearml_upload_data(data,project:str="RFI_mae",name:str='test'):
    ds = Dataset.create(
    dataset_name = name,
    dataset_project = project
    )
    ds.add_files(data)
    ds.upload()
    ds.finalize()  


def clearml_upload_image(images,title:str="VH Original",colorbar:str='VH Intensity'):
    for i in range(len(images)):
        fig1 = plt.figure()
        plt.imshow(images[i,:,:,1],vmin=0,vmax=1,cmap='gray')
        cbar = plt.colorbar()
        cbar.set_label(f'{colorbar}', rotation=270)
        Task.current_task().get_logger().report_matplotlib_figure(
            title = f'{title}\n',
            series = '',
            figure = fig1,
            report_image= True
        )
