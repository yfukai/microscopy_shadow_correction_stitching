#!/usr/bin/env python3
"""
calculate_background.py
determine the dish center position to use backgrounds, and calculate the background by median

"""
import importlib
import sys
import warnings
from datetime import datetime, timedelta
import yaml
import os
from os import path
from time import sleep

import bioformats
import fire
import ipyparallel as ipp
import javabridge
import numpy as np
from numpy import ma as ma
import pandas as pd
import xmltodict
import h5py

from IPython.display import display
from javabridge import jutil
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import filters, io, measure, morphology, transform
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

script_path = path.dirname(path.abspath(__file__))
cziutils_path = path.abspath(path.join(script_path, "../../"))
sys.path.append(cziutils_path)
import cziutils # pylint: disable=import-error
importlib.reload(cziutils)


def read_image(row, reader):
    return reader.read(c=row["C_index"],
                       t=row["T_index"],
                       series=row["S_index"],
                       z=row["Z_index"],
                       rescale=False)


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


def check_validity(x_value, y_value, valid_area, x_edges, y_edges):
    assert np.all(x_edges[1:] > x_edges[:-1])
    assert np.all(y_edges[1:] > y_edges[:-1])
    if x_value >= x_edges[-1] or y_value >= y_edges[-1]:
        return False
    x_bin = np.min(np.where(x_value < x_edges)[0])
    y_bin = np.min(np.where(y_value < y_edges)[0])
    if x_bin > valid_area.shape[0]-1 \
            or y_bin > valid_area.shape[1]-1:
        return False
    return valid_area[x_bin, y_bin]

def with_ipcluster(func):
    def wrapped(*args,**kwargs):
        try:
            proc=


@cziutils.with_javabridge
def calculate_background(filename,
                         output_dir=None,
                         th_factor=3.,
                         valid_ratio_threshold=0.75,
                         intensity_bin_size=25,
                         thumbnail_size=20,
                         quantile=0.001):
    if output_dir is None:
        output_dir = filename[:-4] + "_analyzed"
    params_dict = locals()
    cli = ipp.Client(profile="ipcluster")
    dview = cli[:]
    dview.clear()
    bview = cli.load_balanced_view()
    dview["cziutils_path"] = cziutils_path
    sleep(10)
    dview.block = True
    dview.execute("""
    import javabridge
    import bioformats as bf
    import sys
    print(cziutils_path)
    sys.path.append(cziutils_path)
    import cziutils
    javabridge.start_vm(class_path=bf.JARS)
    """)

    os.makedirs(output_dir, exist_ok=True)

    def savefig(fig, name):
        fig.savefig(path.join(output_dir, name), bbox_inches="tight")

    ############## Load files ################
    meta = cziutils.get_tiled_omexml_metadata(filename)
    reader = cziutils.get_tiled_reader(filename)
    cziutils.summarize_image_size(reader)
    pixel_sizes = cziutils.get_pixel_size(meta)
    assert pixel_sizes[1] == 'µm'
    channels = cziutils.get_channels(meta)
    channel_names = [c["@Fluor"] for c in channels]
    ph_channel_index = [
        j for j, c in enumerate(channels) if "Phase" in c["@Fluor"]
    ][0]
    positions_df = cziutils.get_planes(meta)
    null_indices=positions_df.isnull().any(axis=1)
    params_dict["null_indices"]=positions_df[null_indices].index
    positions_df=positions_df.loc[~null_indices,:]
    positions_df["S_index"] = positions_df["image"]

    ############## Summarize image intensities ################
    dview["filename"] = path.abspath(filename)
    dview.execute("_reader = cziutils.get_tiled_reader(filename)")
    dview["read_image"] = read_image
    dview["summarize_image"] = summarize_image
    sleep(2)
    dview.execute("_reader = cziutils.get_tiled_reader(filename)")
    sleep(1)

    res = bview.map_async(
        lambda row: summarize_image(row, _reader, thumbnail_size, quantile), # pylint: disable=undefined-variable
        [row for _, row in list(positions_df.iterrows())]) 
    res.wait_interactive()
    keys = ["thumbnail", "max", "min", "mean", "median", "stdev"]
    for i, k in enumerate(keys):
        positions_df[k] = [r[i] for r in res.get()]
    display(positions_df)

    ############## Calculate most frequent "standard" mean and stdev for a image ##############
    mean_mode = {}
    stdev_mode = {}
    for iC, grp in positions_df.groupby("C_index"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        c_name = channel_names[iC]
        h, *edges, im = ax.hist2d(grp["mean"],
                                  grp["stdev"],
                                  bins=intensity_bin_size)
        mean_mode[iC],stdev_mode[iC]=\
            [float((edge[x[0]]+edge[x[0]+1])/2.) for edge,x in zip(edges,np.where(h==np.max(h)))]
        ax.plot(mean_mode[iC], stdev_mode[iC], "ro")
        ax.set_xlabel("mean intensity")
        ax.set_ylabel("stdev intensity")
        ax.set_title(c_name)
        savefig(fig, f"1_mean_and_stdev_instensities_{iC}_{c_name}.pdf")

    m, s = mean_mode[ph_channel_index], stdev_mode[ph_channel_index]
    th_low = m - th_factor * s
    th_high = m + th_factor * s
    params_dict.update({
        "channel_names": channel_names,
        "mean_mode": mean_mode,
        "stdev_mode": stdev_mode,
        "ph_th_low": float(th_low),
        "ph_th_high": float(th_high),
    })

    ph_positions_df = positions_df[positions_df["C_index"] ==
                                   ph_channel_index].copy()
    for iS, grp in ph_positions_df.groupby("S_index"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img_mean = grp["thumbnail"].iloc[0]
        axes[0].imshow(img_mean, vmin=th_low, vmax=th_high)
        axes[1].hist(img_mean.flatten(), bins=20, range=(0, 8000))
        axes[1].set_xlabel("intensity")
        axes[1].set_ylabel("freq")
        fig.suptitle("series "+str(iS)
                     +" below th count: "+str(np.sum(img_mean<m-th_factor*s))\
                     +" above th count: "+str(np.sum(img_mean>m+th_factor*s)))
        savefig(fig, f"2_thresholded_thumbnails_{iS}.pdf")

    sigma = 20 / float(pixel_sizes[0])
    params_dict.update({"sigma": sigma})
    dview["threshold_image"] = threshold_image
    sleep(2)
    res = bview.map_async(
        lambda row: threshold_image(row, _reader, sigma, th_low, th_high), # pylint: disable=undefined-variable
        [row for _, row in list(ph_positions_df.iterrows())]) 
    res.wait_interactive()
    ph_positions_df["below_th_count"] = [r[0] for r in res.get()]
    ph_positions_df["above_th_count"] = [r[1] for r in res.get()]
    ph_positions_df.to_hdf(path.join(output_dir, "image_props.hdf5"),
                           "ph_positions")

    ############## Calculate typical thresholded intensities for a image ##############
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    h, x_edges, y_edges, _ = axes[0].hist2d(
        ph_positions_df["below_th_count"],
        ph_positions_df["above_th_count"],
        bins=100)
    th = filters.threshold_otsu(h)
    thresholded = h > th
    clusters = morphology.label(thresholded)
    res = measure.regionprops(
        clusters,
        h,
    )
    mass = [r.mean_intensity * r.area for r in res]
    max_label = res[np.argmax(mass)].label
    valid_area = ndimage.binary_dilation(clusters == max_label,
                                         iterations=3)
    axes[1].imshow(valid_area.T, origin="lower")
    for ax in axes:
        ax.set_xlabel("below threshold counts")
        ax.set_ylabel("above threshold counts")
    axes[0].set_title("raw histogram")
    axes[1].set_title("thresholded")
    fig.tight_layout()
    savefig(fig, f"3_below_above_threshold_counts.pdf")

    ############## judge if the position is valid to calculate background ##############
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ph_positions_df["is_valid"] = ph_positions_df.apply(
        lambda row: check_validity(row["below_th_count"], row[
            "above_th_count"], valid_area, x_edges, y_edges),
        axis=1)
    ax.scatter(ph_positions_df["below_th_count"],
               ph_positions_df["above_th_count"],
               c=ph_positions_df["is_valid"],
               s=1,
               marker="o",
               cmap=plt.get_cmap("viridis"),
               alpha=0.3)
    ax.set_xlabel("below threshold counts")
    ax.set_ylabel("above threshold counts")
    fig.tight_layout()
    savefig(fig, f"4_threshold_results.pdf")

    series_df = pd.DataFrame()
    for Si, grp in ph_positions_df.groupby("S_index"):
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
                },
                index=[Si]))

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

    series_df[
        "is_valid"] = series_df["is_valid_ratio"] > valid_ratio_threshold
    series_df.to_hdf(path.join(output_dir, "image_props.hdf5"),
                     "series")

    valid_series = series_df[series_df["is_valid"]].index
    valid_positions_df = positions_df[positions_df["S_index"].isin(
        valid_series)]
    print("valid_positions:", len(valid_positions_df))

    ############## calclulate backgrounds ##############
    background_df = pd.DataFrame()
    for (iC, iT), grp in \
            tqdm(valid_positions_df.groupby(["C_index", "T_index"])):
        imgs = []
        for i, row in grp.iterrows():
            imgs.append(read_image(row, reader))
        imgs=np.array(imgs)
        lq = np.quantile(imgs, quantile,axis=0)
        hq = np.quantile(imgs, 1.- quantile,axis=0)
        mask=np.logical_or(imgs<lq,imgs>hq)
        imgs_trunc = ma.array(imgs,mask=mask)
        background_df = background_df.append(
            {
                "C_index": iC,
                "T_index": iT,
                "median_img": np.median(imgs, axis=0),
                "mean_img": imgs_trunc.mean(axis=0),
            },
            ignore_index=True)
    background_df["C_index"] = background_df["C_index"].astype(int)
    background_df["T_index"] = background_df["T_index"].astype(int)
    background_df.to_hdf(path.join(output_dir, "background_props.hdf5"),
                          "background")

    ############## check correlation of backgrounds ##############
    for iC, grp in background_df.groupby("C_index"):
        c_name = channel_names[iC]
        for img_key in ["median_img","mean_img"]:
            fig, axes = plt.subplots(1, 6, figsize=(18, 3))
            ps = []
            j = len(grp) // 2
            ims = [grp[img_key].iloc[i] for i in (0, j, -1)]
            ps.append(axes[0].imshow(ims[0]))
            ps.append(axes[1].imshow(ims[1]))
            ps.append(axes[2].imshow(ims[2]))
            ps.append(axes[3].imshow(ims[1] - ims[0]))
            ps.append(axes[4].imshow(ims[1] - ims[2]))
            for p, ax in zip(ps, axes):
                fig.colorbar(p, ax=ax)

            axes[5].plot(ims[0].flatten(), ims[-1].flatten(), ".")
            fig.tight_layout()
            savefig(fig, f"6_background_correlation_{iC}_{c_name}_{img_key}.png")

    ############## summarize and save backgrounds ##############
    with h5py.File(path.join(output_dir, "8_averaged_background.hdf5"),
                   "w") as h5f:
        for iC, grp in background_df.groupby("C_index"):
            for img_key in ["median_img","mean_img"]:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                c_name = channel_names[iC]
                averaged_img = np.mean(grp[img_key], axis=0)
                p = ax.imshow(averaged_img)
                fig.colorbar(p, ax=ax)
                savefig(fig, f"7_averaged_background_{iC}_{c_name}_{img_key}.pdf")

                arrayname = f"background_{int(iC)}_{c_name}_{img_key}"
                filename = path.join(output_dir, arrayname)
                io.imsave(filename + ".tiff",
                          np.round(averaged_img).astype(int))

                ds = h5f.create_dataset(arrayname, data=averaged_img)
                ds.attrs["iC"] = int(iC)
                ds.attrs["c_name"] = c_name
        h5f.attrs["filename"] = filename
        for k, v in params_dict.items():
            print(k,v)
            try:
                h5f.attrs[k] = v
            except TypeError:
                h5f.attrs[k] = np.array(v,dtype="S")

    metadata_xml_name = path.join(output_dir, "metadata.xml")
    with open(metadata_xml_name,"w") as f:
        f.write(meta)


if __name__ == "__main__":
    fire.Fire(calculate_background)
