# CZI shadow correction and stitching

A [Snakemake](https://snakemake.readthedocs.io) workflow for shadow correction and stitching.

## Requirements
- conda and mamba
- snakemake

For installation, follow [this instruction](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

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
3. edit the configuration (option)
```bash
vim config/config.yaml
```
4. run the workflow
```bash
execute_workflow.py N_CORES WORKING_DIRECTORY OUTPUT_DIRECTORY CAMERA_DARK_IMAGE_PATH 
```
- N_CORES ... the number of the cores to use
- WORKING_DIRECTORY ... the directory containing CZI files (can be nested)
- OUTPUT_DIRECTORY ... the directory to output the results (/path/to/output/directory in this case)
- CAMERA_DARK_IMAGE_PATH (optional) ... the path for the dark background image of the camera.
  - path to a image file, read by `skimage.io.imread`
  - path to a directory, with files `a.tiff`, `a.yaml`, `b.tiff`, `b.yaml` ... (the file names can be arbitrary).
    In this case, The YAML files must have keys `"LUT"`, `"binning"`, and `"bit_depth"`. 
    The TIFF file is used if the accompanying YAML file has the same properties as the input CZI file.
  - if not provided, an image filled with zero is used as the background.

5. for each "WORKING_DIRECTORY/aaa/bbb.czi", a directory named "OUTPUT_DIRECTORY/aaa/bbb_analyzed" is created.
   The final artifact is `stitched_image_divide.zarr` (or `stitched_image_subtract.zarr`) in that directory.
