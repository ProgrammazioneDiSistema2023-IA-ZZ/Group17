pub mod onnx_structure;

mod read_proto;

mod read_onnx;
mod convolution_op;
mod relu_op;
mod max_pool_op;

use crate::convolution_op::*;
use crate::max_pool_op::test_max_pool;
use crate::relu_op::*;

use ndarray::{arr2, Axis, concatenate};
mod write_onnx;
use crate::read_onnx::generate_onnx_model;
use crate::read_proto::create_struct_from_proto_file;
use crate::write_onnx::generate_onnx_file;

fn main() {
  let proto_structure = match create_struct_from_proto_file("models/onnx.proto") {
    Ok(proto) => proto,
    Err(err) => panic!("{}", err)
  };

  let mut model = generate_onnx_model("models/squeezenet1.0-8.onnx", &proto_structure);

  let done = generate_onnx_file("models/model_writed.onnx", &mut model);

  println!("{:?}", model);

  test_convolution();
  test_relu();
  test_max_pool();
  test_concat();
}

fn test_concat(){
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