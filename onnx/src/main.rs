pub mod onnx_structure;

use std::io::BufRead;
use ndarray::{arr2, Axis, concatenate};

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

fn main() {
  let mut onnx_file = String::from("models/squeezenet1.0-8.onnx");
  let mut model = generate_onnx_model(&onnx_file, "models/onnx.proto");

  let onnx_generated_file: Vec<&str> = onnx_file.split(".onnx").collect();
  onnx_file = String::from(onnx_generated_file[0]);
  onnx_file.push_str("_generated.onnx");

  let data_0 = read_data_0().unwrap();

  inference(&mut model, data_0);

  generate_onnx_file(&onnx_file, &mut model);

  println!("{:?}", model);
}

fn test_concat() {
  let a = arr2(&[[2., 2.],
    [3., 3.]]);
  assert!(
    concatenate(Axis(0), &[a.view(), a.view()])
      == Ok(arr2(&[[2., 2.],
      [3., 3.],
      [2., 2.],
      [3., 3.]]))
  );
}

fn read_data_0() -> Option<Vec<f32>>{
  use std::fs::File;
  use std::io::BufReader;

  // Specify the file path you want to read
  let file_path = "data_0.txt"; // Replace with your file path

  // Open the file
  let file = File::open(file_path).unwrap();
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