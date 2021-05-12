#!/usr/bin/env python3
import fire
from os import path
from glob import glob
import re
from subprocess import call

SCRIPT_PATH=path.realpath(__file__)
MACRO_PATH=path.abspath(path.join(path.dirname(SCRIPT_PATH),'./stitch_images.ijm'))
print(MACRO_PATH)

def __parse_tile_configuration_filename(filename):
    res=re.search(r"TileConfiguration_t([\d]+)_z([\d]+)_c([\d]+).txt",filename)
    return not res is None, res.groups(0) if not res is None else None

def main(analyzed_dir,imagej_path="ImageJ",show_result=False):
    analyzed_dir=path.abspath(analyzed_dir)
    rescaled_tiff_dir=path.join(analyzed_dir,"rescaled_images_tiff")
    tile_configuration_files=\
        glob(path.join(rescaled_tiff_dir,"TileConfiguration*.txt"))
    res=list(map(__parse_tile_configuration_filename,tile_configuration_files))
    tile_configuration_files=[f \
        for f,r in zip(tile_configuration_files,res) if r[0]]
    print(tile_configuration_files)
    for tile_configuration_file in tile_configuration_files:
        command=f"{imagej_path} -macro '{MACRO_PATH}' '{rescaled_tiff_dir}@{path.basename(tile_configuration_file)}'"
        if not show_result:
            command= command+" -batch"
        print(command)
        call(command,shell=True)

if __name__ == "__main__":
    fire.Fire(main)