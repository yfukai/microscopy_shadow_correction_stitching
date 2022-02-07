params.input_path = "/work/fukai/2021-03-04-timelapse/"
params.background_tiff_path = ""
params.output_path = "/work/fukai/2021-03-04-timelapse_analyzed"

_input_path = new File("${params.input_path}").toURI()
Channel.fromPath("${params.input_path}/**.czi")\
       .map({ 
            relpath=_input_path
              .relativize(it.toFile().toURI())
              .toString().replaceAll(/.czi$/,"_analyzed")
           [it,relpath] 
       }).view({ "${it}" }).into({ input_czi_files })

//profiles {
//    conda {
////        process.conda = "${moduleDir}/envs/image_analysis.yaml"
//        process.conda = "czi_shadow_correction_stitching"
//    }
//}

process exportMetadata {
    cache false
    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"

    input : 
    tuple file(czi_file), val(output_dir) from input_czi_files

    output :
    tuple file(czi_file), val(output_dir), file("metadata.yaml") into cziMetadata

    """
    ${moduleDir}/scripts/a_export_metadata.py \
        ${czi_file} \
        metadata.yaml
    """

}

//process rescaleBackground {
//    publishDir "${output_dir}", pattern: "metadata.yaml", mode: "copy"
//    publishDir "${output_dir}", pattern: "background.npy", mode: "copy"
//
//    input : 
//    tuple file(czi_file), file(output_dir), file("metadata.yaml") from cziMetadata
//
//    output :
//    tuple file(czi_file), file(output_dir), file("metadata.yaml"), file("background.npy") into cziBackground
//
//    """
//    ${moduleDir}/scripts/b_calculate_background.py \
//        ${czi_file} \
//        metadata.yaml \
//        rescaled.zarr \
//        background.npy
//    """
//}

//process rescaleBackground {
//    input : 
//    file czi_file from cziBackground
//
//    output :
//    tuple file(czi_file), file("rescaled.zarr") into cziRescaled
//
//    """
//    ${moduleDir}/scripts/c_rescale_background.py \
//        ${czi_file} \
//        background.npy \
//        rescaled.zarr 
//    """
//}
//
//process stitch {
//
//}
//
//