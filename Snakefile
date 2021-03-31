from glob import glob
from os import path
czi_files = [f[:-4] for f in glob(path.join("**","*.czi"),recursive=True)]

rule all:
  input: 
    expand("{czi_file}.txt",czi_file=czi_files)

rule filename_to_txt:
  input:
    "{czi_file}.czi"
  output:
    "{czi_file}.txt"
  shell:
    "echo {input} > {output}"
    
