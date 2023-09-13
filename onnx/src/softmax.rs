use ndarray::{Array, Axis};
use ndarray::prelude::*;

//OPSET VERSION = 8
fn softmax(mut x: Array2<f32>, axis: Option<usize>) -> Array2<f32> {
  let axis = axis.unwrap_or(1);
  let max_val = x.fold_axis(Axis(axis), f32::NEG_INFINITY, |&acc, &elt| elt.max(acc));
  x -= &max_val.insert_axis(Axis(axis));
  let exp_x = x.mapv(f32::exp);
  let sum_exp_x = exp_x.sum_axis(Axis(axis));
  exp_x / &sum_exp_x.insert_axis(Axis(axis))
}

pub fn test_softmax() {
  // Esempio di utilizzo
  let x = Array::from_shape_vec((2,4), vec![0.,1.,2.,3.,1000.,1001.,1002.,1003.]).unwrap();
  let _x_2 = Array::from_shape_vec((1,3), vec![-1.,0.,1.]).unwrap();
  println!("input: \n{:?}", x);
  let result = softmax(x, None);
  println!("output: \n{:?}", result);
}
