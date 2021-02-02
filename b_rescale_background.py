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
import cziutils

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


def get_camera_acquisition_ROI(meta):
    keys = ['HardwareSetting|ParameterCollection|ImageFrame']
    annotation_dict = cziutils.get_structured_annotation_dict(meta)
    for k in keys:
        try:
            boundary = json.loads(annotation_dict[k])[:4]
            boundary_slice = (slice(boundary[1], boundary[1]+boundary[3]),
                              slice(boundary[0], boundary[0]+boundary[2]))
            return boundary_slice
        except KeyError:
            continue


def rescale_background(filename,
                         analyzed_dir=None,
                         median_sigma=10):
    print(analyzed_dir)
    if analyzed_dir is None:
        analyzed_dir = filename[:-4] + "_analyzed"
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
    binning = cziutils.get_binning(meta)
    binning = np.unique(binning)
    print(filename, binning)
    assert len(binning) == 1
    binning_str = f"{binning[0]}x{binning[0]}"
    boundary = get_camera_acquisition_ROI(meta)
    print("binning: ", binning_str)
    print("ROI: ", boundary)
    params_dict.update({
        "binning": binning_str,
        "ROI": boundary,
    })

    ############## Get corresponding camera dark image ################
    camera_h5f = get_camera_background(meta)
    candidate_keys = [k for k in camera_h5f.keys()
                      if "median_binning"+binning_str in k]
    exposure_time = [float(re.search(r"exposure([\d\.]+)ms", k).groups(0)[0])
                     for k in candidate_keys]
    key = candidate_keys[np.argmax(exposure_time)]
    params_dict.update({
        "camera_dark_hdf5_name": key,
        "camera_dark_key": key,
    })
    camera_dark_img = np.array(camera_h5f[key])[boundary]

    ############## rescale background and save ##############

    output_h5f_name = path.join(
        analyzed_dir, "10_rescaled_background.hdf5")

    with h5py.File(bg_h5f_name, "r") as h5f,\
            h5py.File(output_h5f_name, "w") as h5f_o:
        h5f_o["camera_dark_img"] = camera_dark_img
        for k in h5f.keys():
            fig, axes = plt.subplots(1, 4, figsize=(15, 3))
            bg_img = np.array(h5f[k])
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
