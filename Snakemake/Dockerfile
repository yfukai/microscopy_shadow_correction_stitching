FROM condaforge/mambaforge:latest
LABEL io.github.snakemake.containerized="true"
LABEL io.github.snakemake.conda_env_hash="2e5343f07bab0ecb7798a6386b740070d25c88840f794439033fbe8e00c5036e"

# Step 1: Retrieve conda environments

RUN mkdir -p /conda-envs/5b5515385ee3db23e715506975f6fe75
COPY workflow/envs/conda_env.yaml /conda-envs/5b5515385ee3db23e715506975f6fe75/environment.yaml

# Step 2: Generate conda environments

RUN mamba env create --prefix /conda-envs/5b5515385ee3db23e715506975f6fe75 --file /conda-envs/5b5515385ee3db23e715506975f6fe75/environment.yaml && \
    mamba clean --all -y
