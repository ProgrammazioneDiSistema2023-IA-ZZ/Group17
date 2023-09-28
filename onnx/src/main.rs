pub mod onnx_structure;

use std::io::{Read};
use std::fs::{File};
use protobuf::{Message};

mod read_proto;
mod read_onnx;
mod write_onnx;
mod convolution_op;
mod relu_op;
mod max_pool_op;
mod dropout_op;
mod global_average_pool_op;
mod softmax;
mod model_inference;
mod reshape_op;

use crate::read_onnx::generate_onnx_model;
use crate::model_inference::inference;
use crate::onnx_structure::TensorProto;

fn main() {
  let onnx_file = String::from("models/mnist-8.onnx");
  let input_path = "mnist_data_0.pb";
  let output_path = "mnist_output_0.pb";
  let input_tensor_name = vec!["Input3", "Parameter193"];

  let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");
  //println!("{:?}", model);

  let input_data = read_input_data(input_path).unwrap();
  let output_data = read_input_data(output_path).unwrap();

  inference(&mut model, input_data, input_tensor_name);

  println!("Expected Data: {:?}", output_data);

  /*
  let onnx_generated_file: Vec<&str> = onnx_file.split(".onnx").collect();
  onnx_file = String::from(onnx_generated_file[0]);
  onnx_file.push_str("_generated.onnx");
  generate_onnx_file(&onnx_file, &mut model);
  */

}

fn read_input_data(input_path: &str) -> Option<Vec<f32>>{
  let mut res: Option<Vec<f32>> = None;

  let mut file = File::open(input_path).expect("Cannot open input file");

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer).expect("Error while reading file");

  let parsed_message = TensorProto::parse_from_bytes(&buffer).expect("Error while deserializing the message");

  res = Some(parsed_message.raw_data.clone().unwrap().chunks_exact(4).map(|chunk| u8_to_f32(chunk)).collect());

  res
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  assert_eq!(bytes.len(), 4);
  let mut array: [u8; 4] = Default::default();
  array.copy_from_slice(&bytes);
  f32::from_le_bytes(array)
}