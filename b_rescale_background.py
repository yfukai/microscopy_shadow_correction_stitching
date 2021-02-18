#!/usr/bin/env python3
"""
rescale_background.py
select and subtract the camera fixed background from the estimated background
prerequisites:  
- 8_averaged_background.hdf5 exists for each image in analyzed_dir

"""

import sys
import warnings
from os import path
import json
import re

import fire
import numpy as np
import pandas as pd
import xmltodict
import h5py

from matplotlib import pyplot as plt
from skimage import filters
from skimage.morphology import disk

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

script_path = path.dirname(path.abspath(__file__))
cziutils_path = path.abspath(path.join(script_path, "../../"))
sys.path.append(cziutils_path)
import cziutils # pylint: disable=import-error

camera_dark_directory = path.abspath(
    path.join(script_path, "../../../camera-dark/analyzed/"))
microscope_dict = {
    'HDCamC13440-20CU': "AxioObserver",
    'Axiocam503m': "LSM800"
}


def get_camera_background(meta):
    meta_dict = xmltodict.parse(meta)
    microscope = microscope_dict[meta_dict["OME"]
                                 ["Instrument"]["Detector"]["@Model"]]
    return h5py.File(path.join(camera_dark_directory, microscope+".hdf5"), "r")


def rescale_background(filename,
                         analyzed_dir=None,
                         median_sigma=10,
                         camera_dark_average_method="mean"):
    if analyzed_dir is None:
        analyzed_dir = filename[:-4] + "_analyzed"
    print(analyzed_dir)
    assert camera_dark_average_method in ["mean","median"]
    bg_h5f_name = path.join(analyzed_dir, "8_averaged_background.hdf5")
    metadata_xml_name = path.join(analyzed_dir, "metadata.xml")
    print(bg_h5f_name)
    assert path.isfile(bg_h5f_name)
    assert path.isfile(metadata_xml_name)
    params_dict = locals()
    def savefig(fig, name):
        fig.savefig(path.join(analyzed_dir, name), bbox_inches="tight")

    ############## Load files ################
    with open(metadata_xml_name, "r") as f:
        meta = "".join(f.readlines())

    #id_keys=["camera","binning_str","bit_depth","exposure","LUT"]
    camera_props={
        "binning" : cziutils.get_binning(meta),
        "LUT" : cziutils.get_camera_LUT(meta),
        "bit_depth" : cziutils.get_camera_bits(meta),
    }
    boundary = cziutils.get_camera_roi_slice(meta)
    params_dict.update(camera_props)
    params_dict.update({"boundary" : boundary})
    print(params_dict)

    ############## Get corresponding camera dark image ################
    camera_h5f = get_camera_background(meta)
    candidate_keys=[]
    for k in camera_h5f.keys():
        if all([(np.array_equal(camera_h5f[k].attrs[cp_k], cp_v))
                for cp_k,cp_v in camera_props.items()]):
            candidate_keys.append(k)
    assert len(candidate_keys) > 0

    exposure_time = [float(camera_h5f[k].attrs["exposure"])
                     for k in candidate_keys]
    key = candidate_keys[np.argmax(exposure_time)]
    params_dict.update({
        "camera_dark_hdf5_name": key,
        "camera_dark_key": key,
    })
    camera_dark_img = np.array(camera_h5f[key])
    print(camera_dark_img.shape)
    print(boundary)
    print(params_dict)
    camera_dark_img=camera_dark_img[boundary[1],boundary[0]]

    ############## rescale background and save ##############

    output_h5f_name = path.join(
        analyzed_dir, "10_rescaled_background.hdf5")

    with h5py.File(bg_h5f_name, "r") as h5f,\
            h5py.File(output_h5f_name, "w") as h5f_o:
        h5f_o["camera_dark_img"] = camera_dark_img
        for k in h5f.keys():
            fig, axes = plt.subplots(1, 4, figsize=(15, 3))
            bg_img = np.array(h5f[k])
            assert np.array_equal(bg_img.shape,camera_dark_img.shape)
            normalized = bg_img-camera_dark_img
            normalized_smoothed = filters.median(
                normalized, disk(median_sigma))
            ims = []
            imgs = [camera_dark_img, bg_img,
                    normalized, normalized_smoothed]
            names = ["camera_dark_img", "raw",
                     "normalized", "smoothed"]
            for j, (img, name) in enumerate(zip(imgs, names)):
                if j > 0:
                    h5f_o[name+"_"+k] = img
                ims.append(axes[j].imshow(img))
                axes[j].set_title(name)
            for ax, im in zip(axes, ims):
                fig.colorbar(im, ax=ax)
            savefig(fig, f"9_rescaled_background_{k}.pdf")

        params_dict.update({
            "channel_keys": list(h5f.keys())
        })
        print(params_dict)
        for k, v in params_dict.items():
            try:
                h5f_o.attrs[k] = v
            except TypeError:
                h5f_o.attrs[k] = np.array(v, dtype="S")

if __name__ == "__main__":
    fire.Fire(rescale_background)
