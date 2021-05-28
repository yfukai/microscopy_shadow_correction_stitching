# CZI shadow correction and stitching

A [Snakemake](https://snakemake.readthedocs.io) workflow for shadow correction and stitching.

## Requirements
- conda and mamba
    ```bash
    conda install -n base -c conda-forge mamba
    ```
- fire
    ```
    pip install fire
    ```


<! --
For installation, follow [this instruction](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).
-->

## Usage
1. clone the repository 
    ```bash
    git clone https://github.com/yfukai/czi_shadow_correction_stitching
    cd czi_shadow_correction_stitching
    ```
2. create a directory to output the results.
    ```bash
    mkdir /path/to/output/directory
    ```
3. edit the configuration (optional)
    ```bash
    vim config/config.yaml
    ```
4. choose the method to manage environment
  a. directly create the conda enviroment
    ```bash
    mamba env create -f conda_env.yaml  -n stitching
    ```
  b. use conda environment in Snakemake : add `--conda` option for `./execute_workflow.py`

5. run the workflow
    ```bash
    execute_workflow.py N_CORES WORKING_DIRECTORY OUTPUT_DIRECTORY CAMERA_DARK_IMAGE_PATH --config CONFIG_PATH
    ```
    - N_CORES ... the number of the cores to use
    - WORKING_DIRECTORY ... the directory containing CZI files (can be nested)
    - OUTPUT_DIRECTORY ... the directory to output the results (/path/to/output/directory in this case)
    - CAMERA_DARK_IMAGE_PATH (optional) ... the path for the dark background image of the camera. Can be either of :
      - path to a image file, read by `skimage.io.imread`. The shape should be ((channel count), (Y size of the image, (X size of the image))
      - path to a directory, with files `a.tiff`, `a.yaml`, `b.tiff`, `b.yaml` ... (the file names can be arbitrary).
        In this case, The YAML files must have keys specified `b_rescale_background.match_keys` (default: `"LUT"`, `"binning"`, and `"bit_depth"`). 
        The TIFF file is used if the accompanying YAML file has the same properties as the input CZI file.
      - if not provided, an image filled with zero is used as the background.
    - CONFIG_PATH (optional) ... the path for the configuration file

6. for each "WORKING_DIRECTORY/aaa/bbb.czi", a directory named "OUTPUT_DIRECTORY/aaa/bbb_analyzed" is created.
   The final artifact is `stitched_image_divide.zarr` (or `stitched_image_subtract.zarr`) in that directory.
