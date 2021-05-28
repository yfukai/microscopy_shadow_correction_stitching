FROM condaforge/mambaforge:latest
LABEL io.github.snakemake.containerized="true"
LABEL io.github.snakemake.conda_env_hash="2e5343f07bab0ecb7798a6386b740070d25c88840f794439033fbe8e00c5036e"

# Step 1: Retrieve conda environments

# Conda environment:
#   source: ../../../../../AxioObserver7/ImageData/Fukai/image_analysis/czi_shadow_correction_stitching/workflow/envs/conda_env.yaml
#   prefix: /conda-envs/5b5515385ee3db23e715506975f6fe75
#   name: base
#   channels:
#     - bioconda
#     - defaults
#     - omnia
#     - conda-forge
#   dependencies:
#     - dask=2021.2.0=pyhd3eb1b0_0
#     - dask-core=2021.2.0=pyhd3eb1b0_0
#     - fire=0.3.1=pyh9f0ad1d_0
#     - h5py=2.9.0=py37h7918eee_0
#     - hdf5=1.10.4=hb1b8bf9_0 
#     - mkl=2020.2=256
#     - mkl-service=2.3.0=py37he8ac12f_0
#     - mkl_fft=1.3.0=py37h54f3939_0
#     - mkl_random=1.1.1=py37h0573a6f_0
#     - openjdk=8.0.265=h516909a_0
#     - openssl=1.1.1k=h27cfd23_0
#     - pip=21.0.1=py37h06a4308_0
#     - python=3.7.4=h265db76_1
#     - python-dateutil=2.8.1=pyhd3eb1b0_0
#     - python-libarchive-c=2.9=pyhd3eb1b0_0
#     - python_abi=3.7=1_cp37m
#     - pyyaml=5.4.1=py37h27cfd23_1
#     - readline=7.0=h7b6447c_5
#     - requests=2.25.1=pyhd3eb1b0_0
#     - sqlite=3.31.1=h7b6447c_0
#     - tblib=1.7.0=py_0
#     - wheel=0.36.2=pyhd3eb1b0_0
#     - yaml=0.2.5=h7b6447c_0
#     - zarr=2.8.1=pyhd3eb1b0_0
#     - pip:
#       - ipyparallel==6.2.5
#       - javabridge==1.0.18
#       - m2stitch==0.0.1a0
#       - natsort==7.1.1
#       - numba==0.48.0
#       - numpy==1.20.3
#       - opencv-python-headless==4.5.1.48
#       - pandas==1.2.4
#       - pycziutils==0.3.1
#       - python-bioformats==4.0.4
#       - python-javabridge==4.0.3
#       - scikit-image==0.18.0
#       - scipy==1.6.3
#       - setuptools==50.3.2
#       - snakemake==6.3.0
#       - tables==3.6.1
#       - tqdm==4.60.0
#       - xmltodict==0.12.0
RUN mkdir -p /conda-envs/5b5515385ee3db23e715506975f6fe75
COPY ../../../../../AxioObserver7/ImageData/Fukai/image_analysis/czi_shadow_correction_stitching/workflow/envs/conda_env.yaml /conda-envs/5b5515385ee3db23e715506975f6fe75/environment.yaml

# Step 2: Generate conda environments

RUN mamba env create --prefix /conda-envs/5b5515385ee3db23e715506975f6fe75 --file /conda-envs/5b5515385ee3db23e715506975f6fe75/environment.yaml && \
    mamba clean --all -y
b'96f4009'
snakemake -j24 -d "/mnt/showers_tmp/LSM800/ImageData/KyogoK/2021-05-12/testdata2" --config output_directory="/home/fukai/stitchtest"  camera_dark_path="False" -k --restart-times 5 --configfile config/config_LSM.yaml --containerize
