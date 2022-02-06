params.input_path = "."
params.output_path = "."

input_czi_files = Channel.fromPath($launchDir + "/" + params.input_path)

process exportMetadata {
    input : 
    file czi_file from input_czi_files

    output :
    tuple file(czi_file), file("metadata_yaml.yaml") into fileMetadata

    """
    ${moduleDir}/scripts/a_export_metadata.py \
        --input_czi ${czi_file} \
        --output_metadata_yaml metadata_yaml.yaml
    """
}