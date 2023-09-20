pub mod onnx_structure;

use std::io::BufRead;

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

use crate::read_onnx::generate_onnx_model;
use crate::write_onnx::generate_onnx_file;

use crate::model_inference::inference;
use crate::softmax::test_softmax;

fn main() {
  let mut onnx_file = String::from("models/squeezenet1.0-8.onnx");
  let input_path = "data_0.txt";

  let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");

  let input_data = read_input_data(input_path).unwrap();
  let input_path_split: Vec<&str> = input_path.split(".txt").collect();
  let input_name = String::from(input_path_split[0]);
  inference(&mut model, input_data, input_name.as_str());
  test_softmax();
  /*
  let onnx_generated_file: Vec<&str> = onnx_file.split(".onnx").collect();
  onnx_file = String::from(onnx_generated_file[0]);
  onnx_file.push_str("_generated.onnx");
  generate_onnx_file(&onnx_file, &mut model);
  */

  //println!("{:?}", model);
}

fn read_input_data(input_path: &str) -> Option<Vec<f32>>{
  use std::fs::File;
  use std::io::BufReader;

  // Open the file
  let file = File::open(input_path).unwrap();
  let reader = BufReader::new(file);

  // Create a Vec to store the data
  let mut data: Vec<f32> = Vec::new();

  // Read the data from the file
  for line in reader.lines() {
    let line = line.unwrap();
    // Parse each line as a f32 and push it to the Vec
    if let Ok(value) = line.parse::<f32>() {
      data.push(value);
    } else {
      eprintln!("Error parsing line: {}", line);
    }
  }

  // Now 'data' contains the Vec<f32> with the data from the file
  //println!("{:?}", data);

  Some(data)
}