pub mod onnx_structure;

mod read_proto;

mod read_onnx;
use crate::read_onnx::generate_onnx_model;

fn main() {
  let model = generate_onnx_model("models/squeezenet1.0-8.onnx", "models/onnx.proto");

  println!("{:?}", model);
}