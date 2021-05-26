#!/usr/bin/env python3
"""
calculate_background.py
determine the dish center position to use backgrounds, 
and calculate the background by median

"""
import importlib
import sys
import warnings
import itertools
from datetime import datetime, timedelta
import yaml
import os
from os import path

import bioformats
import fire
import ipyparallel as ipp
import javabridge
import numpy as np
from numpy import ma as ma
from time import sleep
import pandas as pd
import xmltodict
import h5py

from IPython.display import display
from javabridge import jutil
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import filters, io, measure, morphology, transform
io.use_plugin("tifffile")
from tqdm import tqdm

import pycziutils

from utils import with_ipcluster,send_variable,check_ipcluster_variable_defined

def read_image(row, reader):
    return reader.read(c=row["C_index"],
                       t=row["T_index"],
                       series=row["S_index"],
                       z=row["Z_index"],
                       rescale=False)

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

@ipp.require(read_image)
def summarize_image(row, reader, thumbnail_size, quantile):
    import numpy as np
    from skimage import transform
    img = read_image(row, reader)
    lq = np.quantile(img.flatten(), quantile)
    hq = np.quantile(img.flatten(), 1. - quantile)
    img_trunc = img[np.logical_and(img > lq, img < hq)].flatten()
    thumb = transform.rescale(img,
                              thumbnail_size / img.shape[0],
                              preserve_range=True)
    return thumb,np.max(img_trunc),np.min(img_trunc),\
        np.mean(img_trunc),np.median(img_trunc),\
        np.std(img_trunc)


def threshold_image(row, reader, sigma, th_low, th_high):
    import numpy as np
    from skimage import filters
    img = filters.gaussian(read_image(row, reader), sigma, preserve_range=True)
    return np.sum(img < th_low), np.sum(img > th_high)


@pycziutils.with_javabridge
@with_ipcluster
def calculate_background(filename,
                         output_dir,
                         check_validity_channel=False,
                         th_factor=3.,
                         above_threshold_pixel_ratio_max=0.05,
                         below_threshold_pixel_ratio_max=0.05,
                         valid_ratio_threshold=0.4,
                         intensity_bin_size=25,
                         thumbnail_size=20,
                         quantile=0.001,
                         *,
                         ipcluster_nproc=1
                         ):
    params_dict = locals()
    cli = ipp.Client(profile="default")
    dview = cli[:]
    dview.clear()
    bview = cli.load_balanced_view()
    dview.execute("""
    import javabridge
    import bioformats as bf
    import pycziutils
    javabridge.start_vm(class_path=bf.JARS)
    """)

    os.makedirs(output_dir, exist_ok=True)
    log_dir=path.join(output_dir,"calcluate_background_log")
    os.makedirs(log_dir,exist_ok=True)

    def savefig(fig, name):
        fig.savefig(path.join(log_dir, name), bbox_inches="tight")

    ############## Load files ################
    meta = pycziutils.get_tiled_omexml_metadata(filename)
    with open(path.join(output_dir,"metadata.xml"),"w") as f:
        f.write(meta)

    reader = pycziutils.get_tiled_reader(filename)
    _, sizeT, sizeC, sizeX, sizeY, sizeZ = pycziutils.summarize_image_size(reader)

    pixel_sizes = pycziutils.parse_pixel_size(meta)
    assert pixel_sizes[1] == 'µm'
    channels = pycziutils.parse_channels(meta)
    channel_names = [c["@Fluor"] for c in channels]
    print(channel_names)
    if check_validity_channel:
        check_validity_channel_index = [
            j for j, c in enumerate(channels) if check_validity_channel in c["@Fluor"]
        ][0]

    planes_df = pycziutils.parse_planes(meta)
    null_indices=planes_df.isnull().any(axis=1)
    params_dict["null_indices"]=list(planes_df[null_indices].index)
    planes_df=planes_df.loc[~null_indices,:]
    planes_df["S_index"] = planes_df["image"]

    if check_validity_channel:
        ############## Summarize image intensities ################
        send_variable(dview,"filename",path.abspath(filename))
        send_variable(dview,"read_image",read_image)
        send_variable(dview,"summarize_image",summarize_image)
        dview.execute("_reader = pycziutils.get_tiled_reader(filename)")
        check_ipcluster_variable_defined(dview,"_reader",timeout=120)
        sleep(10)
        check_ipcluster_variable_defined(dview,"read_image",timeout=120)
        check_ipcluster_variable_defined(dview,"summarize_image",timeout=120)
    
        @ipp.require(summarize_image)
        def _summarize_image(row):
            return summarize_image(row, _reader, thumbnail_size, quantile) # pylint: disable=undefined-variable
        res = bview.map_async(_summarize_image, 
            [row for _, row in list(planes_df.iterrows())]) 
        res.wait_interactive()
        keys = ["thumbnail", "max", "min", "mean", "median", "stdev"]
        for i, k in enumerate(keys):
            planes_df[k] = [r[i] for r in res.get()]
        display(planes_df)
    
        ############## Calculate most frequent "standard" mean and stdev for a image ##############
        mean_mode = {}
        stdev_mode = {}
        for iC, grp in planes_df.groupby("C_index"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            c_name = channel_names[iC]
            h, *edges, im = ax.hist2d(grp["mean"],
                                      grp["stdev"],
                                      bins=intensity_bin_size)
            mean_mode[iC],stdev_mode[iC]=\
                [float((edge[x[0]]+edge[x[0]+1])/2.) 
                 for edge,x in zip(edges,np.where(h==np.max(h)))]
            ax.plot(mean_mode[iC], stdev_mode[iC], "ro")
            ax.set_xlabel("mean intensity")
            ax.set_ylabel("stdev intensity")
            ax.set_title(c_name)
            savefig(fig, f"1_mean_and_stdev_instensities_{iC}_{c_name}.pdf")
    
        m, s = mean_mode[check_validity_channel_index], stdev_mode[check_validity_channel_index]
        th_low = m - th_factor * s
        th_high = m + th_factor * s
        params_dict.update({
            "channel_names": channel_names,
            "mean_mode": mean_mode,
            "stdev_mode": stdev_mode,
            "ph_th_low": float(th_low),
            "ph_th_high": float(th_high),
        })
    
        ph_planes_df = planes_df[planes_df["C_index"] ==
                                       check_validity_channel_index].copy()
    
        thumbail_output_name="2_thresholded_thumbnail"
        thumbail_output_path=path.join(log_dir,thumbail_output_name)
        os.makedirs(thumbail_output_path,exist_ok=True)
        for iS, grp in ph_planes_df.groupby("S_index"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            img_mean = grp["thumbnail"].iloc[0]
            axes[0].imshow(img_mean, vmin=th_low, vmax=th_high)
            axes[1].hist(img_mean.flatten(), bins=20, range=(0, 8000))
            axes[1].set_xlabel("intensity")
            axes[1].set_ylabel("freq")
            fig.suptitle("series "+str(iS)
                         +" below th count: "+str(np.sum(img_mean<m-th_factor*s))\
                         +" above th count: "+str(np.sum(img_mean>m+th_factor*s)))
            savefig(fig, path.join(thumbail_output_name,
                                   f"2_thresholded_thumbnails_{iS}.pdf"))
            plt.close("all")
    
        sigma = 20 / float(pixel_sizes[0])
        params_dict.update({"sigma": sigma})
        send_variable(dview,"threshold_image",threshold_image)
        res = bview.map_async(
            lambda row: threshold_image(row, _reader, sigma, th_low, th_high), # pylint: disable=undefined-variable
            [row for _, row in list(ph_planes_df.iterrows())]) 
        res.wait_interactive()
        print("ok")
        ph_planes_df["below_th_count"] = [r[0] for r in res.get()]
        ph_planes_df["above_th_count"] = [r[1] for r in res.get()]
        ph_planes_df["below_th_ratio"] = ph_planes_df["below_th_count"] / sizeX / sizeY
        ph_planes_df["above_th_ratio"] = ph_planes_df["above_th_count"] / sizeX / sizeY
        print("ok")
        ph_planes_df.drop("thumbnail",axis=1).to_csv(path.join(log_dir,"ph_planes_df.csv"))
    
        ############## judge if the position is valid to calculate background ##############
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ph_planes_df["is_valid"] = \
            (ph_planes_df["below_th_ratio"] < below_threshold_pixel_ratio_max) & \
            (ph_planes_df["above_th_ratio"] < above_threshold_pixel_ratio_max) 
        ax.scatter(ph_planes_df["below_th_ratio"],
                   ph_planes_df["above_th_ratio"],
                   c=ph_planes_df["is_valid"],
                   s=1,
                   marker="o",
                   cmap=plt.get_cmap("viridis"),
                   alpha=0.3)
        ax.set_xlabel("below threshold ratios")
        ax.set_ylabel("above threshold ratios")
        fig.tight_layout()
        savefig(fig, f"4_threshold_results.pdf")
    
        series_df = pd.DataFrame()
        for Si, grp in ph_planes_df.groupby("S_index"):
            X = grp["X"].iloc[0]
            assert np.all(X == grp["X"])
            Y = grp["Y"].iloc[0]
            assert np.all(Y == grp["Y"])
            series_df = series_df.append(
                pd.DataFrame(
                    {
                        "thumbnail": [np.mean(grp["thumbnail"], axis=0)],
                        "is_valid_ratio": grp["is_valid"].sum() / len(grp),
                        "X": X,
                        "Y": Y
                    }, index=[Si]))
    
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        im = axes[0].scatter(series_df["X"],
                             series_df["Y"],
                             c=series_df["is_valid_ratio"])
        axes[0].set_title("valid_ratio")
        fig.colorbar(im, ax=axes[0])
        axes[1].scatter(series_df["X"],
                        series_df["Y"],
                        c=series_df["is_valid_ratio"] > valid_ratio_threshold)
        axes[1].set_title("thresholded")
        fig.tight_layout()
        savefig(fig, f"5_valid_positions.pdf")
    
        series_df["is_valid"] = series_df["is_valid_ratio"] > valid_ratio_threshold
        series_df.drop("thumbnail",axis=1).to_csv(path.join(log_dir,"series_df.csv"))
        valid_series = series_df[series_df["is_valid"]].index
        planes_df["is_valid"]=planes_df["S_index"].isin(valid_series)
    else:
        planes_df["is_valid"]=True

    valid_planes_df = planes_df[planes_df["is_valid"]]
    print("valid_positions:", len(valid_planes_df))

    planes_df.drop("thumbnail",axis=1,errors="ignore").to_csv(path.join(output_dir,"planes_df.csv"))

    ############## calclulate backgrounds ##############
    #t.c.z.y.x
    median_images=np.empty((sizeT,sizeC,sizeZ,sizeY,sizeX))
    mean_images=np.empty((sizeT,sizeC,sizeZ,sizeY,sizeX))
    median_images[...]=np.nan
    mean_images[...]=np.nan
    print(sizeT)
#    assert np.array_equal(valid_planes_df["T_index"].unique(),np.arange(sizeT))
#    assert np.array_equal(valid_planes_df["C_index"].unique(),np.arange(sizeC))
#    assert np.array_equal(valid_planes_df["Z_index"].unique(),np.arange(sizeZ))

    for (iC, iT, iZ), grp in \
            tqdm(valid_planes_df.groupby(["C_index", "T_index", "Z_index"])):
        imgs = []
        for i, row in grp.iterrows():
            imgs.append(read_image(row, reader))
        imgs=np.array(imgs)
        lq = np.quantile(imgs, quantile,axis=0)
        hq = np.quantile(imgs, 1.- quantile,axis=0)
        mask=np.logical_or(imgs<lq,imgs>hq)
        imgs_trunc = ma.array(imgs,mask=mask)
        median_images[iT,iC,iZ,...]=np.median(imgs, axis=0)
        mean_images[iT,iC,iZ,...]=imgs_trunc.mean(axis=0)


    print("saving background...")
    with h5py.File(path.join(output_dir,"background_per_tile.hdf5"),"w") as h5f:
        h5f.create_dataset("median_images",data=median_images)
        h5f.create_dataset("mean_images",data=mean_images)
#        h5f.attrs["channels"]=channels
        h5f.attrs["dimension_order"]="tczyx"
    print("saved background")

    ############## check correlation of backgrounds ##############
    for iC,iZ in itertools.product(range(sizeC),range(sizeZ)):
        c_name = channel_names[iC]
        for img_key,img in zip(["median","mean"],[median_images,mean_images]):
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            ps = []
            j = sizeT // 2
            ims = [img[i,iC,iZ] for i in (0, j, -1)]
            ps.append(axes[0].imshow(ims[0]))
            ps.append(axes[1].imshow(ims[1]))
            ps.append(axes[2].imshow(ims[2]))
            ps.append(axes[3].imshow(ims[1] - ims[0]))
            ps.append(axes[4].imshow(ims[1] - ims[2]))
            for p, ax in zip(ps, axes):
                fig.colorbar(p, ax=ax)
            axes[5].plot(ims[0].flatten(), ims[-1].flatten(), ".")
            axes[0].set_title("at time 0")
            axes[1].set_title(f"at time {j}")
            axes[2].set_title(f"at time {iT-1}")
            axes[3].set_title(f"diff at time {j} and 0")
            axes[4].set_title(f"diff at time {j} and {iT-1}")
            fig.tight_layout()
            savefig(fig, f"6_background_correlation_C{iC}_{c_name}_Z{iZ}_{img_key}.png")
            plt.close("all")

    ############## summarize and save backgrounds ##############
    background_directory=path.join(output_dir,"averaged_background")
    os.makedirs(background_directory, exist_ok=True)
    for iC,iZ in itertools.product(range(sizeC),range(sizeZ)):
        c_name = channel_names[iC]
        for img_key,img in zip(["median","mean"],[median_images,mean_images]):
            filename=f"{img_key}_C{iC}_{c_name}_Z{iZ}"
            averaged_img = np.nanmean(img[:,iC,iZ], axis=0)
            
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            p = ax.imshow(averaged_img)
            fig.colorbar(p, ax=ax)
            savefig(fig, f"7_time_averaged_background_{filename}.pdf")
            plt.close("all")

            io.imsave(path.join(background_directory,filename + ".tiff"),
                      averaged_img,check_contrast=False)

    params_path=path.join(background_directory,"calculate_background_params.yaml")
    with open(params_path,"w") as f:
        yaml.dump(params_dict,f)
#        for k, v in params_dict.items():
#            print(k,v)
#            try:
#                h5f.attrs[k] = v
#            except TypeError:
#                h5f.attrs[k] = np.array(v,dtype="S")


if __name__ == "__main__":
    try:
        config=snakemake.config["a_calculate_background"]
        print(config)
        calculate_background(snakemake.input["filename"],
                             snakemake.input["output_dir_created"].replace(".created",""),
                             check_validity_channel=config["check_validity_channel"],
                             th_factor=config["th_factor"],
                             above_threshold_pixel_ratio_max=config["above_threshold_pixel_ratio_max"],
                             below_threshold_pixel_ratio_max=config["below_threshold_pixel_ratio_max"],
                             valid_ratio_threshold=config["valid_ratio_threshold"],
                             intensity_bin_size=config["intensity_bin_size"],
                             thumbnail_size=config["thumbnail_size"],
                             quantile=config["quantile"],
                             ipcluster_nproc=snakemake.threads)
    except NameError:
        fire.Fire(calculate_background)
