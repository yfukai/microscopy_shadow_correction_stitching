params.input_path = "/work/fukai/2021-03-04-timelapse/"
params.output_path = "."

Channel.fromPath("${params.input_path}/**.czi").view({ "${it}" }).set({ input_czi_files })

//profiles {
//    conda {
////        process.conda = "${moduleDir}/envs/image_analysis.yaml"
//        process.conda = "czi_shadow_correction_stitching"
//    }
//}

process exportMetadata {
    input : 
    file czi_file from input_czi_files

    output :
    tuple file(czi_file), file("metadata.yaml") into fileMetadata

    """
    ${moduleDir}/scripts/a_export_metadata.py \
        ${czi_file} \
        metadata.yaml
    """
}