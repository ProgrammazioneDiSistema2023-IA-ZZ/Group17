use std::ops::Add;
use ndarray::{Array, Array1, Array4, Dim, Ix4};
use rand::Rng;
use crate::onnx_structure::ModelProto;

use crate::convolution_op::*;
use crate::onnx_structure::*;
use crate::onnx_structure::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use crate::onnx_structure::type_proto::Value;


pub(crate) fn inference(model: &mut ModelProto) {
  for node in &model.graph.node {
    let inputs = &node.input;
    let outputs = &node.output;
    let opertion = match &node.op_type {
      None => { panic!("Operation NOT found for node {}", &node.name.as_ref().unwrap()) }
      Some(op) => { op }
    };

    let mut shape = vec![];

    for input in inputs {
      for inp in &model.graph.input {
        if inp.name.is_some() {
          if inp.name.as_ref().unwrap() == input {
            if inp.type_.value.is_some() {
              match &inp.type_.value.as_ref().unwrap() {
                Value::TensorType(t) => {
                  if t.shape.is_some() {
                    let mut internal_dim = vec![];
                    for el in &t.shape.as_ref().unwrap().dim {
                      if el.value.is_some() {
                        match el.value.as_ref().unwrap() {
                          DimValue(v) => { internal_dim.push(v) }
                          DimParam(_) => {}
                        }
                      }
                    }
                    shape.push(internal_dim);
                  } else {
                    panic!("ERROR WHILE GETTING SHAPE OF NODE {}", &node.name.as_ref().unwrap())
                  }
                }
                Value::SequenceType(_) => {}
                Value::MapType(_) => {}
                Value::OptionalType(_) => {}
                Value::SparseTensorType(_) => {}
              };
            } else {
              panic!("ERROR WHILE GETTING TYPE_ OF NODE {}", &node.name.as_ref().unwrap())
            }
            break;
          }
        }
      }
    }

    let mut raw_data: Vec<Vec<f32>> = vec![];

    for input in inputs {
      for init in &model.graph.initializer {
        if init.name.is_some() {
          if init.name.as_ref().unwrap() == input {
            if init.raw_data.is_some() {
              let mut part_row = vec![];
              for chunk in init.raw_data.as_ref().unwrap().chunks(4) {
                let float32 = u8_to_f32(chunk);
                part_row.push(float32);
              }
              raw_data.push(part_row);
            }
          }
        }
      }
    }

    let mut strides: &Vec<i64> = &Vec::new();
    let mut pads: &Vec<i64> = &Vec::new();;
    let mut kernel_shape: &Vec<i64> = &Vec::new();;

    for attr in &node.attribute {
      if attr.name.is_some() {
        match attr.name.as_ref().unwrap().as_str() {
          "strides" => {
            strides = &attr.ints;
          },
          "pads" => {
            pads = &attr.ints;
          },
          "kernel_shape" => {
            kernel_shape = &attr.ints;
          },
          _ => panic!("ATTRIBUTE NAME NOT FOUND")
        }
      }
    }

    if !strides.is_empty() && !kernel_shape.is_empty() && !pads.is_empty() {
      match opertion.as_str() {
        "Conv" => {
          let (b, c, h, w) = (*shape[0][0] as usize, *shape[0][1] as usize, *shape[0][2] as usize, *shape[0][3] as usize);
          let num_elements = 1 * 1 * h * w * 4;

          // Create a random number generator
          let mut rng = rand::thread_rng();

          // Generate a random Vec<u8>
          let random_data: Vec<u8> = (0..num_elements).map(|_| rng.gen::<u8>()).collect();

          let mut part_row = vec![];
          for chunk in random_data.chunks(4) {
            let float32 = u8_to_f32(chunk);
            part_row.push(float32);
          }

          let input = Array4::from_shape_vec((1, 1, h, w), part_row.clone())
            .unwrap();

          let kernel: Array4<f32> = Array4::from_shape_vec((1, 1, 3, 3), vec![1.; 9])
            .unwrap();

          let bias: Array1<f32> = Array1::from_shape_vec((64), raw_data[1].clone())
            .unwrap();
          /* per lo shape del bias fare una match sul numero delle dimensioni dello shape[2] */

          //println!("{:?}", part_row.clone());
          //println!("{:?}", raw_data[0]);
          //println!("{:?}", raw_data[1]);

          let str: Array1<f32> = strides.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();
          let pad: Array1<f32> = pads.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();

          let conv_layer =
            ConvolutionLayer::new_onnx_tensor_flow(kernel.clone(), Some(bias), Padding::NotSet, None, Some(1), pad, str);
          let output_layer: Array4<f32> = conv_layer.convolve(&input);
          println!("Layer: {:?}", output_layer);
        }
        _ => { panic!("INFERENCE OPERATION NOT FOUND FOR NODE {}", &node.name.as_ref().unwrap()) }
      }
    }
  }
}

fn get_chunk()  {

}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  let mut arr = [0; 4];
  arr.copy_from_slice(bytes);
  f32::from_le_bytes(arr)
}