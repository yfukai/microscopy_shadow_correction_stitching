#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fire
from subprocess import call, check_output
import os
from os import path
import shutil
import yaml
SCRIPT_PATH=path.abspath(__file__)
HOME_PATH = path.expanduser("~")
CACHE_PATH = path.join(HOME_PATH,".shadow_correction_stitching_cache")
os.makedirs(CACHE_PATH,exist_ok=True)
os.environ["SNAKEMAKE_OUTPUT_CACHE"]=CACHE_PATH

def main(n_cores,
         working_directory,
         output_directory,
         config_path="config/config.yaml",
         camera_dark_image_path=False,
         extra_args=""):
    os.chdir(path.dirname(SCRIPT_PATH))
    working_directory=path.abspath(working_directory)
    output_directory=path.abspath(output_directory)
    command = f"snakemake -j{n_cores} -d \"{working_directory}\" " + \
              f"--config output_directory=\"{output_directory}\" " + \
                       f" camera_dark_path=\"{camera_dark_image_path}\" " + \
              f"-k --restart-times 5 --configfile {config_path} --use-conda --cache {extra_args}"
    shutil.copy(path.join(path.dirname(SCRIPT_PATH),"config","config.yaml"),
                path.join(output_directory,"shadow_correction_stitching_snakemake_config.yaml"))
    git_description = check_output(["git", "describe","--always"]).strip()
    with open(path.join(output_directory,
                        "shadow_correction_stitching_config.yaml"),"w") as f:
        yaml.dump({
            "command":command,
            "git_description":git_description,
        },f)
    print(command)
    call(command,shell=True)

if __name__ == "__main__":
    fire.Fire(main)
