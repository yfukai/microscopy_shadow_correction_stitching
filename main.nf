params.input_path_csv = "/home/nplmlab/image_analysis/230512_cellpicker/file_list.txt"
params.output_path = "/mnt/d/Fukai-ImageAnalysis/cell-picker-230512-analyzed"

//_input_path = new File("${params.input_path}").toURI()
//Channel.fromPath("${params.input_path}/**.czi")\
////       .filter( ~/.*210304-HL60-atRAlive-beforelive-01.*/ )
//       .map({ 
//            relpath=_input_path
//              .relativize(it.toFile().toURI())
//              .toString().replaceAll(/.czi$/,"_analyzed")
//           [it,relpath] 
//       }).view({ "${it}" }).into({ input_czi_files })

profiles {
    conda {
//        process.conda = "${moduleDir}/envs/image_analysis.yaml"
        process.conda = "shadow_correction_stitching"
    }
}

process exportMetadata {
    errorStrategy 'retry'
    maxRetries 3

    cache "deep"
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

exportMetadata.subscribe({ println "${it}" })

//process rescaleBackground {
////    errorStrategy 'retry'
////    maxRetries 5
//    maxForks 5
//
//    cache true
//    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"
//    publishDir "${params.output_path}/${output_dir}", pattern: "background.npy", mode: "copy"
////    publishDir "${params.output_path}/${output_dir}", pattern: "rescaled.zarr"
//
//    input : 
//    tuple file(czi_file), file("metadata.yaml"), val(output_dir) from cziMetadata
//
//    output :
//    tuple file("rescaled.zarr"), file("metadata.yaml"), val(output_dir) into rescaledMetadata
//    file "background.npy" 
//
//    """
//    ${moduleDir}/scripts/b_rescale_background.py \
//        ${czi_file} \
//        metadata.yaml \
//        rescaled.zarr \
//        background.npy \
//        -c "${params.background_tiff_path}"
//    """
//}
//
//
//process stitch {
// //   errorStrategy 'ignore'
//    maxForks 20
//
//    publishDir "${params.output_path}/${output_dir}", pattern: "metadata.yaml", mode: "copy"
//    publishDir "${params.output_path}/${output_dir}", pattern: "stitched.zarr", mode: "copy"
//    publishDir "${params.output_path}/${output_dir}", pattern: "stitching_result.csv", mode: "copy"
//
//    input :
//    tuple file("rescaled.zarr"), file("metadata.yaml"), val(output_dir) from rescaledMetadata
//
//    output :
//    tuple file("stitched.zarr"), file("metadata.yaml"), val(output_dir) into stitchedMetadata
//    file "stitching_result.csv" 
//
//    """
//    echo ${output_dir}
//    ${moduleDir}/scripts/c_stitch.py \
//        rescaled.zarr \
//        metadata.yaml \
//        stitched.zarr \
//        stitching_result.csv
//    """
//}
//
//process report {
//    publishDir "${params.output_path}/${output_dir}", pattern: "report", mode: "copy"
//
//    input :
//    tuple file("stitched.zarr"), file("metadata.yaml"), val(output_dir) from stitchedMetadata
//
//    output : 
//    val(output_dir) into reported
//    file "report" 
//
//    """
//    ${moduleDir}/scripts/d_report.py \
//        stitched.zarr \
//        metadata.yaml \
//        report
//    """
//}
//
//reported.subscribe({ println "${it}" })
//