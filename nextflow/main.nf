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
    cache true
    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"

    input : 
    tuple file(czi_file), val(output_dir) from input_czi_files

    output :
    tuple file(czi_file), file("metadata.yaml"), val(output_dir) into cziMetadata

    """
    ${moduleDir}/scripts/a_export_metadata.py \
        ${czi_file} \
        metadata.yaml
    """

}

process rescaleBackground {
    cache true
    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"
    publishDir "${params.output_path}/${output_dir}", pattern: "background.npy", mode: "copy"
    publishDir "${params.output_path}/${output_dir}", pattern: "rescaled.zarr"

    input : 
    tuple file(czi_file), file("metadata.yaml"), val(output_dir) from cziMetadata

    output :
    tuple file("rescaled.zarr"), file("metadata.yaml"), val(output_dir) into rescaledMetadata

    """
    ${moduleDir}/scripts/b_rescale_background.py \
        ${czi_file} \
        metadata.yaml \
        rescaled.zarr \
        background.npy
    """
}

//process stitch {
//    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"
//    publishDir "${params.output_path}/${output_dir}", pattern: "stitched.zarr", mode: "copy"
//
//    input :
//    tuple file("rescaled.zarr"), file("metadata.yaml"), val(output_dir) from rescaledMetadata
//
//    output :
//    tuple file("stitched.zarr"), file("metadata.yaml"), val(output_dir) to stitchedMetadata
//
//    """
//    ${moduleDir}/scripts/c_stitch.py \
//        rescaled.zarr \
//        metadata.yaml \
//        stitched.zarr
//    """
//}