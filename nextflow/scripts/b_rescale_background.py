#!/usr/bin/env python3
#%%
from aicsimageio import AICSImage
import click
import yaml
import numpy as np
from matplotlib import pyplot as plt
from dask import array as da
from skimage import filters, transform, io
from skimage.morphology import disk


"""
calculate_background.py
determine the dish center position to use backgrounds, 
and calculate the background by median

"""
def scaled_median_filter(im2d,scale,size):
    assert im2d.ndim == 2
    shape = im2d.shape
    im2d = np.array(im2d, dtype=np.float32)
    im2d = transform.rescale(im2d, 
        scale,
        anti_aliasing=False,preserve_range=True)
    im2d = filters.median(im2d,
            disk(size))
    return transform.resize(im2d,shape,
                preserve_range=True)

@click.command()
@click.argument("input_czi", type=click.Path(exists=True))
@click.argument("metadata_yaml", type=click.Path(exists=True))
@click.argument("output_zarr", type=click.Path())
@click.argument("background_npy", type=click.Path())
@click.option("--camera_background_tiff", "-c", type=click.Path(exists=True),default=None)
@click.option("--mode", "-m", 
             type=click.Choice(['divide', 'subtract'], case_sensitive=False), 
             default="subtract")
def main(
    input_czi,
    metadata_yaml,
    output_zarr,
    background_npy,
    camera_background_tiff,
    mode,
    choosepos_target_channel="Phase",
    choosepos_median_filter_scaling=0.1,
    choosepos_median_filter_size=10,
    choosepos_stdev_quantile=0.25,
    choosepos_stdev_factor=5,
    choosepos_pixel_ratio_threshold=0.05,
    background_gaussian_filter_sigma=10,
):
    print(" loading data ")
    aics_image = AICSImage(input_czi, reconstruct_mosaic=False)
    with open(metadata_yaml, "r") as f:
        metadata=yaml.safe_load(f)
    if camera_background_tiff is not None:
        camera_background = io.imread(camera_background_tiff)
    else:
        camera_background = np.zeros(aics_image.shape[-2:],dtype=np.float32)

    image=aics_image.get_image_dask_data("MTCZYX",)\
                    .rechunk([1,1,1,1,*aics_image.shape[-2:]])
    channel_names=metadata["channel_names"]
    choosepos_target_index=channel_names.index(choosepos_target_channel)
    
    time_median_image=np.median(
        image[:,:,choosepos_target_index,:,:,:]-camera_background,
        axis=1)
    
   
    print(" applying filter ")
    filtered_image=da.from_array(
        [[scaled_median_filter(time_median_image[m,z],
            choosepos_median_filter_scaling,
            choosepos_median_filter_size)
           for z in range(time_median_image.shape[1])]
           for m in range(time_median_image.shape[0])]).compute()
    median_image=np.median(filtered_image,axis=0)
    stds=np.std(filtered_image,axis=(1,2,3))
    threshold = np.quantile(
        stds,choosepos_stdev_quantile)*\
        choosepos_stdev_factor

    diff_median_image=(np.abs(filtered_image-median_image) > threshold)
    diff_median_ratio=diff_median_image.sum(axis=(1,2,3))\
                        /np.product(diff_median_image.shape[1:]).astype(float)
    
    ##indices = np.argsort(diff_median_ratio)
    ##for i in range(diff_median_image.shape[0]):
    ##    j=indices[i]
    ##    plt.subplot(121)
    ##    plt.imshow(image[j,0])
    ##    plt.subplot(122)
    ##    plt.imshow(diff_median_image[j,0])
    ##    plt.suptitle(diff_median_ratio[j])
    ##    plt.show()
    #%%
    
    print(" finding position finished ")
    used_mosaic_index=diff_median_ratio<choosepos_pixel_ratio_threshold
    metadata["used_mosaic_index"]=list(map(int,np.where(used_mosaic_index)[0]))
    print(metadata)
    with open(metadata_yaml+"_2.yaml", "w") as f:
        yaml.safe_dump(metadata,f)


    ############## calclulate backgrounds ##############

    print(" calculating background ")
    mosaic_median_image=np.median(
        image[used_mosaic_index,:,:,:,:,:]-camera_background,axis=0)
    flatfield=np.median(mosaic_median_image,axis=0)
    flatfield=da.from_array([[
        filters.gaussian( flatfield[c,z,:,:],
                          sigma=background_gaussian_filter_sigma )
        for z in range(flatfield.shape[1])]
        for c in range(flatfield.shape[0])]
    ).compute()

    darkfield=np.tile(
        camera_background,
        list(flatfield.shape[:-2])+[1,1])
    print(" calculating background finished ")

    ############## summarize and save backgrounds ##############
    np.save(background_npy,[flatfield,darkfield])

#    background_directory = path.join(output_dir, "averaged_background")
#    os.makedirs(background_directory, exist_ok=True)
#    for iC, iZ in itertools.product(range(sizeC), range(sizeZ)):
#        c_name = channel_names[iC]
#        for img_key, img in zip(["median", "mean"], [median_images, mean_images]):
#            filename = f"{img_key}_C{iC}_{c_name}_Z{iZ}"
#            averaged_img = np.nanmean(img[:, iC, iZ], axis=0)
#
#            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#            p = ax.imshow(averaged_img)
#            fig.colorbar(p, ax=ax)
#            savefig(fig, f"7_time_averaged_background_{filename}.pdf")
#            plt.close("all")
#
#            io.imsave(
#                path.join(background_directory, filename + ".tiff"),
#                averaged_img,
#                check_contrast=False,
#            )

    ############## rescale by background ##############

    print(" rescaling background ")
    if mode == "divide":
        rescaled_image=(image-darkfield)/flatfield
    elif mode == "subtract":
        rescaled_image = image-darkfield-flatfield
    else:
        assert False, "unknown mode"
    rescaled_image.to_zarr(output_zarr,overwrite=True)
    print(" rescaling background finished ")

if __name__ == "__main__":
    main()

# %%
import numpy as np
import zarr
# %%
