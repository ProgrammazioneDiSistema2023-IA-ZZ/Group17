pub mod onnx_structure;

mod read_proto;

mod read_onnx;
mod convolution_op;
mod relu_op;
mod max_pool_op;
mod dropout_op;
mod global_average_pool_op;
mod softmax;

use crate::read_onnx::generate_onnx_model;
use crate::convolution_op::*;
use crate::max_pool_op::test_max_pool;
use crate::relu_op::*;
use crate::dropout_op::test_dropout;
use crate::global_average_pool_op::test_global_average_pool;

use ndarray::{arr2, Axis, concatenate};
use crate::softmax::test_softmax;


fn main() {
  let model = generate_onnx_model("models/squeezenet1.0-8.onnx", "models/onnx.proto");

  println!("{:?}", model);

  test_convolution();
  test_relu();
  test_max_pool();
  test_concat();
  test_dropout();
  test_global_average_pool();
  test_softmax();
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