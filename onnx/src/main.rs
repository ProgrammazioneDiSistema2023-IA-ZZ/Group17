pub mod onnx_structure;

mod read_proto;

mod read_onnx;
mod convolution_op;

use crate::read_onnx::generate_onnx_model;
use crate::convolution_op::*;

fn main() {
  let model = generate_onnx_model("models/squeezenet1.0-8.onnx", "models/onnx.proto");

  println!("{:?}", model);
  test();
}