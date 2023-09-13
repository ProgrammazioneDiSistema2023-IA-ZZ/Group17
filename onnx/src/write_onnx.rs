use std::fs::File;
use protobuf::{CodedOutputStream, Message};
use crate::onnx_structure::ModelProto;

pub fn generate_onnx_file(onnx_file_path: &str, model_proto: &mut ModelProto) -> bool {
    let mut file = File::create(onnx_file_path).unwrap();

    /*let mut output = CodedOutputStream::new(&mut file);
    model_proto.write_to_with_cached_sizes(&mut output).expect("ERROR");*/

    let mut output_stram = CodedOutputStream::new(&mut file);
    match model_proto.write_to_with_cached_sizes(&mut output_stram) {
        Ok(_) => { println!("MODEL WRITED ON FILE CORRECTLY"); true }
        Err(_) => { println!("ERRO WHILE WRITING MODEL ON FILE"); false}
    }
}