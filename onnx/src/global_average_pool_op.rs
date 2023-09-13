use ndarray::{Array, Array4, Axis};

//OPSET VERSION. 1 channel out
fn global_average_pool(x: Array4<f32>) -> Array4<f32> {
  let mut sum: f32 = x.iter().sum();
  let counter = x.len();

  sum = sum / counter as f32;

  Array4::from_shape_vec((x.len_of(Axis(0)), x.len_of(Axis(1)), 1, 1), vec![sum]).unwrap()
}

pub fn test_global_average_pool() {
  //(batch size, channels out, height, width)
  let input = Array::from_shape_vec(
    (1, 1, 4, 4),
    vec![1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]
  )
    .unwrap();
  println!("input: {:?}", input);
  let output = global_average_pool(input);
  println!("output: {:?}", output);
}
