from os import path
from aicsimageio import AICSImage
from matplotlib import pyplot as plt
import basicpy
import numpy as np
from dask import array as da
from tqdm import tqdm
import os
from skimage import transform, filters, morphology

def basicpy_shading_calculation(img_data : da.Array, camera_darkfield = None):
    if camera_darkfield is None:
        camera_darkfield = np.zeros(img_data.shape[-2:],dtype=np.float32)
    else:
        assert camera_darkfield.shape == img_data.shape[-2:]
    shading_fitting_results = []
    basic = basicpy.BaSiC(get_darkfield=False,smoothness_flatfield=1.)
    n_C = img_data.shape[0]
    for c in range(n_C):
        channel_img_data = img_data[c,:,0]-camera_darkfield[np.newaxis,:,:]
        basic.fit(channel_img_data.compute())
        shading_fitting_results.append({
            "flatfield" : basic.flatfield,
            "darkfield" : basic.darkfield,
            "baseline" : basic.baseline,
        })
    return shading_fitting_results

def plot_shading_correction_result(
        shading_fitting_results:dict,
        ncols:int=5
    ):
    n_C = len(shading_fitting_results)
    nrows = ((n_C-1)//ncols + 1)
    fig = plt.figure(figsize=(3*ncols, 2.5*nrows *3))
    subfigs = fig.subfigures(nrows,ncols)
    for c, sfig in zip(range(n_C), np.ravel(subfigs)):
        res = shading_fitting_results[c]
        sfig.suptitle("")
        ax1, ax2, ax3 = sfig.subplots(3,1)
        im = ax1.imshow(res["flatfield"])
        ax1.axis("off")
        ax1.set_title("flatfield")
        fig.colorbar(im,ax=ax1)
        im = ax2.imshow(res["darkfield"])
        ax2.axis("off")
        ax2.set_title("darkfield")
        fig.colorbar(im,ax=ax2)

        ax3.plot(res["baseline"])
        ax2.set_title("baseline")
    return fig
    
def scaled_filter(im2d,scale,fn,anti_aliasing=True):
    shape = im2d.shape
    im2d = np.array(im2d, dtype=np.float32)
    im2d = transform.rescale(im2d, 
        scale,
        anti_aliasing=anti_aliasing,
        preserve_range=True)
    im2d = fn(im2d)
    return transform.resize(im2d,shape,
                preserve_range=True)

def local_subtraction(img, scaling=0.1, median_disk_size=4):
    def median_filter(im):
        return filters.median(
                       im,morphology.disk(median_disk_size)
                    )
    def subtract_smoothed(im):
        _im = im[tuple([0]*(im.ndim-2))]
        return im-scaled_filter(_im, scaling, median_filter, anti_aliasing=False)[tuple([np.newaxis]*(im.ndim-2))]

    return img.rechunk([1]*(img.ndim-2)+list(img.shape[-2:])).map_blocks(
        subtract_smoothed, dtype = img.dtype)

def correct_shading(img_data,
                    shading_fitting_results,
                    mode = "additive", 
                    local_subtraction_channels=[]):
    if mode == "additive":
        bg = np.array([np.median(res["baseline"]) * res["flatfield"] 
                       for res in shading_fitting_results])
        img_data = img_data - bg[:,np.newaxis,np.newaxis,:,:]
    elif mode == "multiplicative":
        bg = np.array(res["flatfield"] for res in shading_fitting_results)
        img_data = img_data / bg[:,np.newaxis,np.newaxis,:,:]
    else:
        raise ValueError("mode value is wrong")

    if len(local_subtraction_channels) > 0:
        img_data[local_subtraction_channels, ... ] = \
            local_subtraction(img_data[local_subtraction_channels, ... ])

    return img_data