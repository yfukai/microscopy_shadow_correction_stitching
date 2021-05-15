#! /usr/bin/env python3
import numpy as np
import os
from os import path
import re
import h5py
import zarr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import fire
from tqdm import tqdm
from dask import bag as db
from dask.diagnostics import ProgressBar
from skimage.segmentation import relabel_sequential

def row_to_indices(dimension_order,**pos):
    nonspecified_keys=[k for k in dimension_order if not k in pos.keys()]
    indices=tuple([int(pos[k]) if k in pos.keys() else slice(None)
                  for k in dimension_order])
    return indices,nonspecified_keys

def get_channel_ind(c_name,channels):
    ind=[j for j,c in enumerate(channels) if c_name in str(c)]
    if not len(ind)==1:
        print(c_name,channels)
        raise AssertionError()
    return ind[0]

def reorder_labels(mask,offset=0):
    values, counts = np.unique(mask.flatten(), return_counts=True)
    base_index=values[np.argmax(counts)] # find the most frequent label (=background)
    if base_index!=0:
        mask[mask==0]=np.max(mask)+1
        mask[mask==base_index]=0
    mask2,_,_=relabel_sequential(mask,offset)
    return mask2,np.max(mask2)+1

def parse_stitching_result(filename):
    pattern=r"rescaled_t([\d]+)_row([\d]+)_col([\d]+)_color([\d]+)\.tiff; ; \(([\-\d\.]+), ([\-\d\.]+)\)"
    names=["t","row","col","color","pos_x","pos_y"]
    parsed_result=[]
    with open(filename,"r") as f:
        for line in f.readlines():
            res=re.search(pattern,line)
            if not res is None:
                parsed_result.append(dict(zip(names,res.groups())))
    stitching_result_df=pd.DataFrame.from_records(parsed_result)    
    for k in names[:4]:
        stitching_result_df[k]=stitching_result_df[k].astype(np.int32)
    for k in names[4:]:
        stitching_result_df[k]=stitching_result_df[k].astype(np.float64)
    return stitching_result_df

#def is_touching_boundary(label,submask):
#    border_image=np.zeros(submask.shape,dtype=np.bool)
#    border_image[0,:]=True
#    border_image[-1,:]=True
#    border_image[:,0]=True
#    border_image[:,-1]=True
#    return np.any(np.logical_and(submask,border_image))

def stitching(stitching_result_txt,
              rescaled_image,
              dimension_order,
              channels,
              planes_df,
              mask=None,
              *,
              savefig_dir=False,
              rescale_method="divide"):
    
    stitching_result_df=parse_stitching_result(stitching_result_txt)
    if savefig_dir:
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        im=axes[0].scatter(stitching_result_df["pos_x"],
                    stitching_result_df["pos_y"],c=
                    stitching_result_df["row"],
                    cmap=plt.cm.Paired)
        fig.colorbar(im,ax=axes[0])
        im=axes[1].scatter(stitching_result_df["pos_x"],
                    stitching_result_df["pos_y"],c=
                    stitching_result_df["col"],
                    cmap=plt.cm.Paired)
        fig.colorbar(im,ax=axes[1])
        fig.savefig(path.join(savefig_dir,"stitching1_row_col_configuration.pdf"))
    
    planes_df["X_index2"]=(planes_df["X_index"]+1).astype(np.int32)
    planes_df["Y_index2"]=(planes_df["Y_index"]+1).astype(np.int32)
    stitching_result_df2=pd.merge(stitching_result_df,planes_df,
                                  left_on=("col","row","color"),
                                  right_on=("X_index2","Y_index2","C_index"),
                                  how="left")
    print(len(stitching_result_df),len(stitching_result_df2))
    stitching_result_df2["pos_x2"]=stitching_result_df2["pos_x"]\
                                   .apply(np.round).astype(np.int32)
    stitching_result_df2["pos_y2"]=stitching_result_df2["pos_y"]\
                                   .apply(np.round).astype(np.int32)
    stitching_result_df2["pos_x2"]=\
        stitching_result_df2["pos_x2"]-stitching_result_df2["pos_x2"].min()
    stitching_result_df2["pos_y2"]=\
        stitching_result_df2["pos_y2"]-stitching_result_df2["pos_y2"].min()
    
    x_index_pos=dimension_order.index("x")
    y_index_pos=dimension_order.index("y")
    x_size=rescaled_image.shape[x_index_pos]
    y_size=rescaled_image.shape[y_index_pos]
    x_width=stitching_result_df2["pos_x2"].max()+x_size
    y_width=stitching_result_df2["pos_y2"].max()+y_size

    if savefig_dir:
        fig,ax=plt.subplots(1,1)
        ax.plot(stitching_result_df2["pos_x"]-stitching_result_df2["X"],label="x")
        ax.plot(stitching_result_df2["pos_y"]-stitching_result_df2["Y"],label="y")
        ax.set_xlabel("series")
        ax.set_ylabel("difference b/w original and assigned position (px)")
        fig.savefig(path.join(savefig_dir,"stitching2_position_difference.pdf"))
    
    stitched_image=np.zeros((len(channels),x_width,y_width),dtype=rescaled_image.dtype)
    if not mask is None:
        stitched_mask=np.zeros((x_width,y_width),dtype=np.int64)
    else:
        stitched_mask=None
    
    mask_label_offset=1
    border_image=np.zeros((x_size,y_size),dtype=np.bool)
    border_image[0,:]=True
    border_image[-1,:]=True
    border_image[:,0]=True
    border_image[:,-1]=True

    for _,row in tqdm(stitching_result_df2.iterrows(),
                      total=len(stitching_result_df2)):
        x2=int(row["pos_x2"])
        y2=int(row["pos_y2"])
        window=(slice(x2,x2+x_size),slice(y2,y2+y_size))

        indices1,dimension_order3=row_to_indices(dimension_order,s=row["image"])
#        print(dimension_order3)
        indices2,_=row_to_indices(
            dimension_order.replace("c",""),s=row["image"])
        series_image=rescaled_image[indices1]
        target_dimension_order=list("cxy")
        src=list(map(lambda x : dimension_order3.index(x),"cxy"))
        dst=list(map(lambda x : target_dimension_order.index(x),"cxy"))
        series_image=np.moveaxis(series_image,src,dst)
        stitched_image[tuple([slice(None)]+list(window))]=series_image

        if not mask is None:
#            print("mask")
            series_mask=mask[indices2]
            src=list(map(lambda x : list(filter(
                lambda x : x!="c",dimension_order3)).index(x),"xy"))
            dst=list(map(lambda x : list(filter(
                lambda x : x!="c",target_dimension_order)).index(x),"xy"))
#            print(src,dst,list(filter(lambda x : x!="c",dimension_order)),list(filter(
#                lambda x : x!="c",target_dimension_order)))
            series_mask=np.moveaxis(series_mask,src,dst)
            #remove all overlapping mask
            
            labels=np.setdiff1d(
                    np.unique(stitched_mask[window]),
                    np.unique(stitched_mask[window][border_image]))
            labels=labels[labels>0]
            stitched_mask[np.isin(stitched_mask,labels)]=0
            
            series_mask,mask_label_offset=reorder_labels(
                series_mask,mask_label_offset)
            
            labels=np.unique(series_mask[border_image])
            labels=labels[labels>0]
            series_mask[np.isin(series_mask,labels)]=0
            stitched_mask[window]+=series_mask
    #stitched_mask=reorder_labels(stitched_mask)
    return stitched_image,stitched_mask

def process_stitching(output_dir,
                      stitching_csv_path=None,
                      rescale_methods=["divide"]):
    
    if stitching_csv_path is None:
        stitching_csv_path=path.join(output_dir,
            "rescaled_images_for_stitching_tiff",
            "TileConfiguration.registered.txt")
    rescaled_image_directory=path.join(output_dir,"rescaled_images")
    stitching_log_directory=path.join(output_dir,"stitching_log")
    os.makedirs(stitching_log_directory,exist_ok=True)

    params_dict=locals()

    planes_df=pd.read_hdf(output_dir,"planes_df2.csv")


    for rescale_method in rescale_methods:
        rescaled_image_key=f"rescaled_image_{rescale_method}"
        rescaled_image_directory2=path.join(rescaled_image_directory,rescaled_image_key)
        assert path.isdir(rescaled_image_directory2)
        output_zarr_path=path.join(output_dir,"stitched_image_{rescale_method}.zarr")
        output_zarr=zarr.open(output_zarr_path,mode="r")
        channels=list(map(lambda x:x.decode("ascii"),channels))
    
    
    t_z_indices=[(t,z) for (t,z),_ in planes_df.groupby(["T_index","Z_index"])]
    
    def execute_stitching_for_single_plane(args):
        t,z=args
        stitching_result_txt=path.join(output_dir,
            f"rescaled_images_tiff/TileConfiguration_t{t+1:03d}_z{z+1:03d}_c001.registered.txt")
        stitching_log_directory2=path.join(stitching_log_directory,f"t{t+1:03d}_z{z+1:03d}")
        os.makedirs(stitching_log_directory2,exist_ok=True)
        indices,dimension_order2=row_to_indices(dimension_order,t=t,z=z)
        indices_mask,_=row_to_indices(dimension_order.replace("c",""),t=t,z=z)

        with h5py.File(rescaled_image_path) as h5f:
            ds=h5f[rescaled_image_key]
            rescaled_image=np.array(ds[indices])
        mask_z5f=z5py.File(mask_zarr_path)
        mask=mask_z5f["mask"][indices_mask]
#        test_ds=output_zarr.create_dataset("test",shape=(1,1),dtype=np.int8)
#        print(channels)
#        test_ds.attrs["channels"]=channels

        stitched_image,stitched_mask=stitching(stitching_result_txt,
                  rescaled_image,
                  "".join(dimension_order2),
                  channels,
                  planes_df[(planes_df["T_index"]==t)&
                            (planes_df["Z_index"]==z)],
                  mask,
                  savefig_dir=stitching_log_directory2,
                  rescale_method=rescale_method)
        assert np.all(stitched_mask)>=0
        image_ds=output_zarr.create_dataset(
            f"image_t{t+1:03d}_z{z+1:03d}",
            data=stitched_image,
            chunks=(1,2048,2048))
        image_ds.attrs["channels"]=list(channels)
        mask_ds=output_zarr.create_dataset(
            f"mask_t{t+1:03d}_z{z+1:03d}",
            data=stitched_mask,
            chunks=(2048,2048))
    
    with ProgressBar(): 
        db.from_sequence(t_z_indices)\
          .map(execute_stitching_for_single_plane)\
          .compute(num_workers=20)
