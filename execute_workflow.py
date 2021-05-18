#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fire
from subprocess import call
import os
SCRIPT_PATH=os.path.abspath(__file__)

def main(working_directory,
         output_directory,
         camera_dark_image_path,
         n_cores):
    os.chdir(SCRIPT_PATH)
    command = 
    call()

if __name__ == "__main__":
    fire.Fire(main)
