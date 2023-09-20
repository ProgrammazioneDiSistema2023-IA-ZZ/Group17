use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4, Axis, concatenate};
use rand::Rng;
use crate::onnx_structure::ModelProto;

use crate::convolution_op::{ConvolutionLayer as ConvLayerConv, Padding as PadConv};
use crate::dropout_op::dropout;
use crate::global_average_pool_op::global_average_pool;
use crate::onnx_structure::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use crate::onnx_structure::type_proto::Value;
use crate::relu_op::relu;
use crate::max_pool_op::{ConvolutionLayer as ConvLayerMaxPool, Padding as PadMaxPool};
use crate::softmax::softmax;


pub(crate) fn inference(model: &mut ModelProto, data_0: Vec<f32>) {
  let mut hashmap_outputs_to_inputs: HashMap<String, Array4<f32>> = HashMap::new();

  for node in &model.graph.node {
    let inputs = &node.input;

    let get_wire_data = contains_worb(inputs);

    let outputs = &node.output;
    let operation = match &node.op_type {
      None => { panic!("Operation NOT found for node {}", &node.name.as_ref().unwrap()) }
      Some(op) => { op }
    };

    let mut shape = vec![];
    let mut raw_data: Vec<Vec<f32>> = vec![];

    if operation == "Conv" && get_wire_data {
      /* GET SHAPES OF RAW DATA*/
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

      /* GET RAW DATA */
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
    }

    /*if raw_data.len() != inputs.len() {
      /* NEED TO GET RESULTS CALCULATES IN THE PRECEDENTS STEPS */
      for input in inputs {
        raw_data.push()
      }
    }*/


    let mut strides: &Vec<i64> = &Vec::new();
    let mut pads: &Vec<i64> = &Vec::new();
    let mut kernel_shape: &Vec<i64> = &Vec::new();
    let mut axis: &Vec<i64> = &Vec::new();
    let mut ratio: f32 = Default::default();

    if operation != "Relu" && operation != "GlobalAveragePool" && operation != "Softmax" {
      /* GET STRIDES, PADS AND KERNEL SHAPE */
      for attr in &node.attribute {
        if attr.name.is_some() {
          match attr.name.as_ref().unwrap().as_str() {
            "strides" => {
              strides = &attr.ints;
            }
            "pads" => {
              pads = &attr.ints;
            }
            "kernel_shape" => {
              kernel_shape = &attr.ints;
            }
            "axis" => {
              axis = &attr.ints;
            }
            "ratio" => {
              ratio = attr.f.unwrap()
            }
            _ => panic!("ATTRIBUTE NAME NOT FOUND")
          }
        }
      }
    }

    match operation.as_str() {
      "Conv" => {
        let (mk, gk, hk, wk) = (*shape[1][0] as usize, *shape[1][1] as usize, *shape[1][2] as usize, *shape[1][3] as usize);

        let input;
        /* TODO: non è detto che l'input principale sita data_0 per tutti i modelli!! */
        if inputs[0] == "data_0" {
          let (bs, cs, hs, ws) = (*shape[0][0] as usize, *shape[0][1] as usize, *shape[0][2] as usize, *shape[0][3] as usize);
          input = Array4::from_shape_vec((bs, cs, hs, ws), data_0.clone())
            .unwrap();
        } else {
          input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());
        }

        let kernel: Array4<f32> = Array4::from_shape_vec((mk, gk, hk, wk), raw_data[0].clone())  /* RAW_DATA[0] -> W */
          .unwrap();
        let bias: Array1<f32> = Array1::from_shape_vec(*shape[2][0] as usize, raw_data[1].clone()) /* RAW_DATA[1] -> B */
          .unwrap();
        let str: Array1<f32> = strides.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();
        let pad: Array1<f32> = pads.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();

        let conv_layer = ConvLayerConv::new_onnx_tensor_flow(kernel.clone(), Some(bias), PadConv::NotSet, None, Some(1), pad, str);
        let output_layer: Array4<f32> = conv_layer.convolve(&input);

        println!("Conv: {:?}", output_layer);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer);
      }
      "Relu" => {
        let input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());

        let output_layer: Array4<f32> = relu(&input);

        println!("Relu: {:?}", output_layer);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer);
      }
      "MaxPool" => {
        let input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());

        let (mk, gk) = (kernel_shape[0] as usize, kernel_shape[1] as usize);

        let kernel: Array2<i32> = Array2::zeros((mk, gk)); /* TODO: CONTROLLARE */
        let str: Array1<f32> = strides.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();
        let pad: Array1<f32> = pads.iter().map(|&x| x as f32).collect::<Vec<f32>>().into();

        let conv_layer = ConvLayerMaxPool::new(PadMaxPool::Valid, pad, kernel, Some(0), str);
        let output_layer: Array4<f32> = conv_layer.max_pool(&input);

        println!("MaxPool: {:?}", output_layer);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer);
      }
      "Concat" => {
        let input_1 = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());
        let input_2 = Array4::from(hashmap_outputs_to_inputs.get(inputs[1].as_str()).unwrap().clone());

        /* TODO: Non è detto ci sia sempre e solo un axis in posizione 0 */
        let output_layer: Array4<f32> = concatenate(Axis(axis[0] as usize), &[input_1.view(), input_2.view()]).unwrap();

        println!("Concatenate: {:?}", output_layer);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer);
      }
      "Dropout" => {
        let input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());

        let output_layer = dropout(input, Some(ratio), None, false, false);

        println!("Dropout: {:?}", output_layer.0);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer.0);

        if output_layer.1.is_some() {
          println!("Mask: {:?}", output_layer.1.unwrap());

          /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
          //hashmap_outputs_to_inputs.insert(outputs[1].clone(), output_layer.1.unwrap());
        }
      }
      "GlobalAveragePool" => {
        let input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());

        let output_layer = global_average_pool(input);

        println!("GlobalAveragePool: {:?}", output_layer);

        /* TODO: non è detto che ci sia sempre e solo un output quindi controllare */
        hashmap_outputs_to_inputs.insert(outputs[0].clone(), output_layer);
      }
      "SoftMax" => {
        let input = Array4::from(hashmap_outputs_to_inputs.get(inputs[0].as_str()).unwrap().clone());

        let result = softmax(input, None);

        println!("SoftMax: {:?}", result);
      }
      _ => { panic!("INFERENCE OPERATION NOT FOUND FOR NODE {}", &node.name.as_ref().unwrap()) }
    }
  }
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  let mut arr = [0; 4];
  arr.copy_from_slice(bytes);
  f32::from_le_bytes(arr)
}

fn contains_worb(strings: &Vec<String>) -> bool {
  for string in strings.iter() {
    if string.contains('w') || string.contains('b') {
      return true;
    }
  }
  false
}