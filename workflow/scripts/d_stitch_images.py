#!/usr/bin/env python3
import fire
from os import path
from glob import glob
import re
import numpy as np
import pandas as pd
from subprocess import Popen
from utils import parse_stitching_result

SCRIPT_PATH=path.realpath(__file__)
MACRO_PATH=path.abspath(path.join(path.dirname(SCRIPT_PATH),'./stitch_images.ijm'))
print(MACRO_PATH)

def __parse_tile_configuration_filename(filename):
    res=re.search(r"TileConfiguration_t([\d]+)_z([\d]+)_c([\d]+).txt",filename)
    return not res is None, res.groups(0) if not res is None else None

def main(output_dir,imagej_path="ImageJ",timeout=60*60*12,
         show_result=False,only_first_timepoint=False,
         n_procs=1):
    output_dir=path.abspath(output_dir)
    rescaled_tiff_dir=path.join(output_dir,"rescaled_images_for_stitching_tiff")
    stitching_log_directory=path.join(output_dir,"stitching_log")

    tile_configuration_files=\
        glob(path.join(rescaled_tiff_dir,"TileConfiguration*.txt"))
    res=list(map(__parse_tile_configuration_filename,tile_configuration_files))
    tile_configuration_files=[f \
        for f,r in zip(tile_configuration_files,res) if r[0]]
    print(tile_configuration_files)
    if only_first_timepoint:
        tile_configuration_files=[tile_configuration_files[0]]

    procs=[]
    for j,tile_configuration_file in enumerate(tile_configuration_files):
        command=f"{imagej_path} -macro '{MACRO_PATH}' '{rescaled_tiff_dir}@{path.basename(tile_configuration_file)}'"
        if not show_result:
            command= command+" -batch"
        print(command)
        proc=Popen(command,shell=True)
        procs.append(proc)
        if (j+1)%n_procs==0:
            for proc in procs:
                proc.wait(timeout=timeout)
            procs.clear()
    for proc in procs:
        proc.wait(timeout=timeout)
    

    stitching_df=pd.DataFrame()
    for tile_configuration_file in tile_configuration_files:
        tile_configuration_file2=tile_configuration_file.replace(".txt",".registered.txt")
        df=parse_stitching_result(tile_configuration_file2)
        df["tile_configuration_file"]=tile_configuration_file2
        stitching_df=stitching_df.append(df)
#    print(stitching_df)

    stitching_df.to_csv(path.join(output_dir,"stitching_result_raw.csv"))
#    count=np.max(stitching_df[["row","col","pos_x"]].groupby(["row","col"]).count()["pos_x"].reset_index())
#    print("coiunt",count)
    stitching_df2_median=stitching_df[["row","col","pos_x","pos_y"]]\
                    .groupby(["row","col"]).median().reset_index()
    stitching_df2_std=stitching_df[["row","col","pos_x","pos_y"]]\
                    .groupby(["row","col"]).std().reset_index()
    stitching_df2_std=stitching_df2_std.fillna(0)
    stitching_df2=pd.merge(stitching_df2_median,stitching_df2_std,
                           on=["row","col"],suffixes=["_median","_std"])
    stitching_df2.to_csv(path.join(output_dir,"stitching_result_sumamrized.csv"))
    

if __name__ == "__main__":
    try:
        main(path.dirname(snakemake.input["output_dir_created"]),
                          only_first_timepoint=snakemake.config["stitching_only_first_timepoint"],
                          n_procs=snakemake.threads)
    except NameError:
        fire.Fire(main)
