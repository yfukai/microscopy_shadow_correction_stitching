#%%
!pip install xmltodict pycziutils
#%%
import xmltodict
from os import path
import numpy as np
import javabridge
import bioformats 
import pycziutils
import functools
import yaml
javabridge.start_vm(class_path=bioformats.JARS)
myloglevel = "ERROR"  # user string argument for logLevel.
rootLoggerName = javabridge.get_static_field(
    "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;"
)
rootLogger = javabridge.static_call(
    "org/slf4j/LoggerFactory",
    "getLogger",
    "(Ljava/lang/String;)Lorg/slf4j/Logger;",
    rootLoggerName,
)
logLevel = javabridge.get_static_field(
    "ch/qos/logback/classic/Level",
    myloglevel,
    "Lch/qos/logback/classic/Level;",
)
javabridge.call(
    rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel
)


#%%
match_keys_ops = {
    "LUT": lambda meta: list(pycziutils.parse_camera_LUT(meta)),
    "binning": pycziutils.parse_binning,
    "bit_depth": pycziutils.parse_camera_bits,
}


def wrap_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def get_camera_props(
    meta,
    channel,
    wf_match_keys=["LUT", "binning", "bit_depth"],
    cf_match_keys=[],
):
    meta_dict = xmltodict.parse(meta)
    detectors = wrap_list(meta_dict["OME"]["Instrument"]["Detector"])
    acq_mode = channel["@AcquisitionMode"]
    detector_id = channel["DetectorSettings"]["@ID"]
    detector = [d for d in detectors if d["@ID"] == detector_id]
    assert len(detector) == 1
    detector = detector[0]

    if acq_mode == "WideField":
        prop_dict = {k: match_keys_ops[k](meta) for k in wf_match_keys}
        boundary = pycziutils.parse_camera_roi_slice(meta)
    elif acq_mode == "LaserScanningConfocalMicroscopy":
        prop_dict = {k: match_keys_ops[k](meta) for k in cf_match_keys}
        boundary = (slice(None), slice(None))
    else:
        raise RuntimeError(f"Acquisition mode {acq_mode} not supported")

    if detector["@Model"] != "":
        prop_dict["camera"] = detector["@Model"]
    else:
        prop_dict["camera"] = detector["@ID"]
    return prop_dict, boundary
#%%
def main(filename,camera_dark_path):
    meta = pycziutils.get_tiled_omexml_metadata(filename)
    channels = pycziutils.parse_channels(meta)
    camera_propss = [
        get_camera_props(meta, channel)
        for channel in channels
    ]
    # id_keys=["camera","binning_str","bit_depth","exposure","LUT"]
    params_dict={}
    params_dict.update({"camera_propss": [p[0] for p in camera_propss]})
    params_dict.update(
        {"boundary": [[[b.start, b.stop] for b in p[1]] for p in camera_propss]}
    )
    print(params_dict)

    camera_dark_img = []
    for channel, (camera_props, boundary) in zip(channels, camera_propss):
        candidate_files = []
        propss = []
        exposures = []
        path_pattern = path.join(
            camera_dark_path, f"mean_*{camera_props['camera']}*.tiff"
        )
#        print(path_pattern)
        image_files = glob(path_pattern)
        assert len(image_files) > 0
        for f in image_files:
#            print(f)
            with open(f.replace(".tiff", ".yaml")) as f2:
                props = yaml.safe_load(f2)
#            print([(props[k], camera_props[k])
#                   for k in camera_props.keys()])
            #XXX dirty impl!
            def custom_array_equal(a1,a2):
                try:
                    return np.array_equal(a1,a2, equal_nan=True)
                except:
                    return np.array_equal(a1,a2)

            if all(
                [
                    custom_array_equal(props[k], camera_props[k],)
                    for k in camera_props.keys()
                ]
            ):
                candidate_files.append(f)
                propss.append(props)
                exposures.append(props["exposure"])
#                print("")
        dark_image_file = candidate_files[np.argmax(exposures)]
        props = propss[np.argmax(exposures)]
        del props["meta"]
        params_dict.update(
            {
                "camera_dark_image_file": path.abspath(dark_image_file),
                "camera_dark_image_props": props,
                "meta":meta
            }
        )
    return params_dict
 
#%%
from glob import glob
from tqdm import tqdm
dark_image_file_props={}
camera_dark_path="/mnt/showers/AxioObserver7/ImageData/Fukai/camera-dark/analyzed"
for f in tqdm(glob("/work/fukai/2021-03-04-timelapse/**/*.czi",recursive=True)):
    if path.isfile(f):
        print(f)
        params_dict=main(f,camera_dark_path)
        dark_image_file_props[f]=params_dict

# %%
files=np.unique(list(map(lambda x:x["camera_dark_image_file"],
    dark_image_file_props.values()))).tolist()
assert len(files)==1
dark_image_file=files[0]
print(dark_image_file)
# %%
from skimage.io import imread, imsave
img=imread(dark_image_file)

# %%
boundaries=np.unique(list(map(lambda x:x["boundary"],
    dark_image_file_props.values()))).tolist()
# %%
b=np.concatenate(np.concatenate(boundaries))
by=b[1::2]
bx=b[::2]
assert np.all(by==by[0])
assert np.all(bx==bx[0])
# %%
img2=img[by[0][0]:by[0][1],bx[0][0]:bx[0][1]]
print(img2.shape)
imsave("/work/fukai/2021-03-04-timelapse/camera_dark_image.tiff",img2)
# %%
