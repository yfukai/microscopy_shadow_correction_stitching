params.input_path = "/work/fukai/2021-03-04-timelapse/"
params.background_path = ""
params.output_path = "."

Channel.fromPath("${params.input_path}/**.czi")\
       .view({ "${it}" })\
       .into({ input_czi_files1, input_czi_files2, input_czi_files3 })

//input_czi_files3.map { it.toString() }.into({ input_czi_files })

//profiles {
//    conda {
////        process.conda = "${moduleDir}/envs/image_analysis.yaml"
//        process.conda = "czi_shadow_correction_stitching"
//    }
//}

process exportMetadata {
    input : 
    file czi_file from input_czi_files1

    output :
    tuple file(czi_file), file("metadata.yaml") into cziMetadata

    """
    ${moduleDir}/scripts/a_export_metadata.py \
        ${czi_file} \
        metadata.yaml
    """
}

process calcBackground {
    input : 
    file czi_file from input_czi_files2

    output :
    tuple file(czi_file), file("background.npy") into cziBackground

    """
    ${moduleDir}/scripts/b_calculate_background.py \
        ${czi_file} \
        background.npy
    """
}

process rescaleBackground {
    input : 
    file czi_file from cziBackground

    output :
    tuple file(czi_file), file("rescaled.zarr") into cziRescaled

    """
    ${moduleDir}/scripts/c_rescale_background.py \
        ${czi_file} \
        background.npy \
        rescaled.zarr 
    """
}

process stitch {

}

