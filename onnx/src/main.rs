pub mod onnx_structure;

mod read_proto;

mod read_onnx;
mod convolution_op;
mod relu_op;
mod max_pool_op;

use crate::read_onnx::generate_onnx_model;
use crate::convolution_op::*;
use crate::max_pool_op::test_max_pool;
use crate::relu_op::*;

use ndarray::{arr2, Axis, concatenate};

fn main() {
  let model = generate_onnx_model("models/squeezenet1.0-8.onnx", "models/onnx.proto");

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