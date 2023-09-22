use std::collections::HashMap;
use std::default;
use std::hash::Hash;
use ndarray::{Array1, Array2, Array3, Array4, Axis, concatenate};
use crate::onnx_structure::{ModelProto, NodeProto, TensorProto, ValueInfoProto};

use crate::convolution_op::{ConvolutionLayer as ConvLayerConv, Padding as PadConv};
use crate::dropout_op::dropout;
use crate::global_average_pool_op::global_average_pool;
use crate::onnx_structure::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use crate::onnx_structure::type_proto::Value;
use crate::relu_op::relu;
use crate::max_pool_op::{ConvolutionLayer as ConvLayerMaxPool, Padding as PadMaxPool};
use crate::reshape_op::reshape;
use crate::softmax::softmax;


pub(crate) fn inference(model: &mut ModelProto, input_data: Vec<f32>, input_tensor_name: Vec<&str>) {
  let mut hashmap_outputs_to_inputs: HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)> = HashMap::new();

  manage_input_data(&mut hashmap_outputs_to_inputs, model, input_data, input_tensor_name);

  for node in &model.graph.node {
    let operation = match &node.op_type {
      None => { panic!("Operation {:?} NOT found for node {}", &node.op_type, &node.name.as_ref().unwrap()) }
      Some(op) => { op }
    };

    match operation.as_str() {
      "Conv" => convolution_op(&mut hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
      "Relu" => relu_op(&mut hashmap_outputs_to_inputs, node),
      "MaxPool" => max_pool_op(&mut hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
      "Concat" => concatenate_op(&mut hashmap_outputs_to_inputs, node),
      "Dropout" => drop_out_op(&mut hashmap_outputs_to_inputs, node),
      "GlobalAveragePool" => global_average_pool_op(&mut hashmap_outputs_to_inputs, node),
      "Softmax" => softmax_op(&mut hashmap_outputs_to_inputs, node),
      "Reshape" => reshape_op(&mut hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
      "Add" => add_op(&mut hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
      "MatMul" => mul_op(&mut hashmap_outputs_to_inputs, node),
      _ => { panic!("INFERENCE OPERATION '{}' NOT FOUND FOR NODE {}", operation.as_str(), &node.name.as_ref().unwrap()) }
    }
  }
}

fn manage_input_data(hashmap_outputs_to_inputs: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, model: &ModelProto, input_data: Vec<f32>, input_tensor_name: Vec<&str>){
  for input_name in input_tensor_name{
    if !already_into_initializer(&model.graph.initializer, input_name) {
      let dims: Vec<&i64> = search_input_data_shape(&model.graph.input, input_name);
      let array = Array4::from_shape_vec((*dims[0] as usize, *dims[1] as usize, *dims[2] as usize, *dims[3] as usize), input_data.clone()).unwrap();
      hashmap_outputs_to_inputs.insert(input_name.to_string(), (None, Some(array)));
    }
  }
}

fn convolution_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) {
  let input_image = match output_container.get(&node.input[0]).clone() {
    Some(image) => image.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };
  let kernel = match output_container.get(&node.input[1]).clone() {
    Some(ker) => ker.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };
  let mut bias: Option<Array1<f32>> = None;
  if node.input.len() > 2usize {
    let (_, _, _, arr1, _) = get_stored_tensor_for_convolution(2, node, model_inputs, model_initializers);
    bias = Some(arr1.unwrap());
  }

  let mut strides: Array1<f32> = Default::default();
  let mut pads: Array1<f32> = Default::default();
  let mut auto_pad: PadConv = PadConv::Valid;
  let mut group: Option<i32> = Some(1);
  let mut dilations: Option<Array2<i32>> = None;
  for attr in &node.attribute {
    if attr.name.is_some() {
      match attr.name.as_ref().unwrap().as_str() {
        "auto_pad" => match std::str::from_utf8(&attr.s.clone().unwrap()).unwrap() {
          "SAME_UPPER" => auto_pad = PadConv::SameUpper,
          "SAME_LOWER" => auto_pad = PadConv::SameLower,
          "VALID" => auto_pad = PadConv::Valid,
          "NOT_SET" => auto_pad = PadConv::NotSet,
          _ => panic!("Convolution Auto Pad specified not found: {}", std::str::from_utf8(&attr.s.clone().unwrap()).unwrap())
        },
        "dilations" => dilations = Some(Array2::from_shape_vec((1, 2), vec![attr.ints[0]as i32, attr.ints[1] as i32]).unwrap()),
        "group" => group = Some(attr.i.unwrap() as i32),
        "kernel_shape" => {},
        "pads" => pads = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        "strides" => strides = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        _ => panic!("ATTRIBUTE NAME FOR CONVOLUTION NOT FOUND, {}", attr.name.as_ref().unwrap().as_str())
      }
    }
  }
  let conv_layer = ConvLayerConv::new_onnx_tensor_flow(kernel.clone(), bias, auto_pad, dilations, group, pads, strides);
  let output_layer: Array4<f32> = conv_layer.convolve(&input_image);

  dbg!("Conv: {:?}", output_layer.clone());
  //println!("Conv: {:?}", output_layer.clone());
  println!("Convolve, done!");
  output_container.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn relu_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
  let output_layer: Array4<f32> = relu(&input);

  dbg!("Relu: {:?}", output_layer.clone());
  println!("Relu, done!");
  output_container.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn max_pool_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>){
  let input_image = match output_container.get(&node.input[0]).clone() {
    Some(image) => image.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };

  let mut kernel_shape: Array2<i32> = Default::default();
  let mut strides: Array1<f32> = Default::default();
  let mut pads: Array1<f32> = Default::default();
  let mut auto_pad: PadMaxPool = PadMaxPool::Valid;
  let mut storage_order: Option<i32> = None;
  for attr in &node.attribute {
    if attr.name.is_some() {
      match attr.name.as_ref().unwrap().as_str() {
        "auto_pad" => match std::str::from_utf8(&attr.s.clone().unwrap()).unwrap() {
          "SAME_UPPER" => auto_pad = PadMaxPool::SameUpper,
          "SAME_LOWER" => auto_pad = PadMaxPool::SameLower,
          "VALID" => auto_pad = PadMaxPool::Valid,
          "NOTSET" => auto_pad = PadMaxPool::NotSet,
          _ => panic!("MaxPool Auto Pad specified not found: {}", std::str::from_utf8(&attr.s.clone().unwrap()).unwrap())
        },
        "kernel_shape" => kernel_shape = Array2::zeros((attr.ints[0].clone() as usize, attr.ints[1].clone() as usize)),
        "pads" => pads = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        "storage_order" => storage_order = Some(attr.i.unwrap() as i32),
        "strides" => strides = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        _ => panic!("ATTRIBUTE NAME FOR MAXPOOL NOT FOUND, {}", attr.name.as_ref().unwrap().as_str())
      }
    }
  }
  let conv_layer = ConvLayerMaxPool::new(auto_pad, pads, kernel_shape, storage_order, strides);
  let output_layer: Array4<f32> = conv_layer.max_pool(&input_image);

  dbg!("MaxPool: {:?}", output_layer.clone());
  println!("MaxPool, done!");
  output_container.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn concatenate_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input_1 = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
  let input_2 = Array4::from(output_container.get(node.input[1].as_str()).unwrap().1.clone().unwrap().clone());
  let mut axis = 1;

  for attr in &node.attribute {
    if attr.name.is_some() {
      axis = match attr.name.as_ref().unwrap().as_str() {
        "axis" => attr.i.unwrap(),
        _ => panic!("ATTRIBUTE NAME FOR CONCATENATE NOT FOUND, {}", attr.name.as_ref().unwrap().as_str())
      };
    }
  }
  let output_layer: Array4<f32> = concatenate(Axis(axis as usize), &[input_1.view(), input_2.view()]).unwrap();

  //dbg!("Concatenate: {:?}", output_layer);
  println!("Concatenate, done!");
  output_container.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn drop_out_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  let mut ratio: Option<f32> = None;
  for attr in &node.attribute {
    if attr.name.is_some() {
      ratio = match attr.name.as_ref().unwrap().as_str() {
        "ratio" => Some(attr.f.unwrap()),
        _ => panic!("ATTRIBUTE NAME FOR DROP OUT NOT FOUND, {}", attr.name.as_ref().unwrap().as_str())
      };
    }
  }
  let output_layer = dropout(input, ratio, None, false, false);

  /*
   if output_layer.1.is_some() {
     println!("Mask: {:?}", output_layer.1.unwrap());
     hashmap_outputs_to_inputs.insert(outputs[1].clone(), output_layer.1.unwrap());
   }
  */

  dbg!("Dropout: {:?}", output_layer.0.clone());
  println!("Dropout, done!");

  output_container.insert(node.output[0].clone(), (None, Some(output_layer.0)));
}

fn global_average_pool_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  let output_layer = global_average_pool(input);

  //dbg!("GlobalAveragePool: {:?}", output_layer);
  println!("GlobalAveragePool, done!");

  output_container.insert(node.output[0].clone(), (None, Some(output_layer)));
}

fn softmax_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  let result = softmax(input, None);

  println!("Softmax, done!");
  dbg!(result.clone());

  /*
  let mut i = 0;
  while i < 1000 {
    if result[[0, i]] == 1.0 {
      println!("Class {}-nth predicted.", i);
      break;
    }
    i += 1;
  }
  */
}

fn reshape_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>){
  let mut data: Array4<f32> = Default::default();
  if already_into_initializer(model_initializers, node.input[0].as_str()){
    let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
    data = arr4.unwrap();
  } else{
    data = Array4::from(output_container.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
  }
  
  let mut shape: Array1<i64> = Default::default();
  if already_into_initializer(model_initializers, node.input[1].as_str()){
    let (_, _, _, _, arr1) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
    shape = arr1.unwrap();
  } else{
    panic!("Cannot get Shape for Reshape operation");
  }

  let output_layer: Array2<f32> = reshape(data, shape, None);

  dbg!("Reshape: {:?}", output_layer.clone());
  println!("Reshape, done!");
  output_container.insert(node.output[0].clone(), (Some(output_layer), None));
}

fn add_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>){
  let mut input_1_arr_4: Array4<f32> = Default::default();
  let mut input_1_arr_2: Array2<f32> = Default::default();
  let mut input_2_arr_3: Array3<f32> = Default::default();
  let mut input_2_arr_2: Array2<f32> = Default::default();

  if already_into_initializer(model_initializers, node.input[0].as_str()){
    let (arr4, _, arr2, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
    match arr4{
      Some(arr4) => input_1_arr_4 = arr4,
      None => {
        match arr2{
          Some(arr2) => input_1_arr_2 =  arr2,
          None => panic!("Cannot get input 1 for Add operation from initializers")
        }
      }
    };
  }
  else{
    match output_container.get(node.input[0].as_str()).unwrap().0.clone(){
      Some(arr2) => input_1_arr_2 = arr2,
      None => {
        match output_container.get(node.input[0].as_str()).unwrap().1.clone(){
          Some(arr4) => input_1_arr_4 = arr4,
          None => panic!("Cannot get input 1 for Add operation from hashmap input/output")
        }
      }
    }
  }

  if already_into_initializer(model_initializers, node.input[1].as_str()){
    let (_, arr3, arr2, _, _) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
    match arr3 {
      Some(arr3) => input_2_arr_3 = arr3,
      None => {
        match arr2 {
          Some(arr2) => input_2_arr_2 = arr2,
          None =>  panic!("Cannot get input 2 for Add operation from initializes")
        }
      }
    }
  } else{
    panic!("Cannot get input 2 for Add operation");
  }

  let mut output_layer_2: Array2<f32> = Default::default();
  let mut output_layer_4: Array4<f32> = Default::default();
  if input_1_arr_4.len() > 0{
    output_layer_4 = input_1_arr_4+input_2_arr_3;
    dbg!("Add: {:?}", output_layer_4.clone());
    println!("Add, done!");
    output_container.insert(node.output[0].clone(), (None, Some(output_layer_4)));
  }else{
    output_layer_2 = input_1_arr_2+input_2_arr_2;
    dbg!("Add: {:?}", output_layer_2.clone());
    println!("Add, done!");
    output_container.insert(node.output[0].clone(), (Some(output_layer_2), None));
  }
}

fn mul_op(output_container: &mut HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>, node: &NodeProto){
  let input_1 = Array2::from(output_container.get(node.input[0].as_str()).unwrap().0.clone().unwrap());
  let input_2 = Array2::from(output_container.get(node.input[1].as_str()).unwrap().0.clone().unwrap());

  let output_layer: Array2<f32> = input_1.dot(&input_2);

  dbg!("MatMul: {:?}", output_layer.clone());
  println!("MatMul, done!");
  output_container.insert(node.output[0].clone(), (Some(output_layer), None));
}

fn get_stored_tensor_for_convolution(i: usize, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) -> (Option<Array4<f32>>, Option<Array3<f32>>, Option<Array2<f32>>, Option<Array1<f32>>, Option<Array1<i64>>){
  let mut shape = search_input_data_shape(model_inputs, &node.input[i]);

  let mut raw_data: Vec<f32> = vec![];
  let mut raw_data_i64: Vec<i64> = vec![];
  for init in model_initializers {
    if init.name.is_some() {
      if init.name.as_ref().unwrap() == &node.input[i] {
        if init.raw_data.is_some() {
          for chunk in init.raw_data.as_ref().unwrap().chunks(4) {
            let float32 = u8_to_f32(chunk);
            raw_data.push(float32);
          }
        }else if init.float_data.len() > 0 {
          for float_value in &init.float_data{
            raw_data.push(*float_value);
          }
        }else if init.int64_data.len() > 0{
          for int_value in &init.int64_data{
            raw_data_i64.push(*int_value);
          }
        }
      }
    }
  }

  if shape.len() == 4{
    (Some(Array4::from_shape_vec((*shape[0] as usize, *shape[1] as usize, *shape[2] as usize, *shape[3] as usize), raw_data).unwrap()), None, None, None, None)
  }else if shape.len() == 3{
    (None, Some(Array3::from_shape_vec((*shape[0] as usize, *shape[1] as usize, *shape[2] as usize), raw_data).unwrap()), None, None, None)
  }else if shape.len() == 2{
    (None, None, Some(Array2::from_shape_vec((*shape[0] as usize, *shape[1] as usize), raw_data).unwrap()), None, None)
  }else{
    if raw_data.len() > 0 {
      (None, None, None, Some(Array1::from_shape_vec((*shape[0] as usize), raw_data).unwrap()), None)
    }else {
      (None, None, None, None, Some(Array1::from_shape_vec((*shape[0] as usize), raw_data_i64).unwrap()))
    }
  }
}

fn u8_to_f32(bytes: &[u8]) -> f32 {
  let mut arr = [0; 4];
  arr.copy_from_slice(bytes);
  f32::from_le_bytes(arr)
}

fn search_input_data_shape<'a>(model_inputs: &'a Vec<ValueInfoProto>, input_name: &str) -> Vec<&'a i64> {
  let mut shape = vec![];
  for inp in model_inputs {
    if inp.name.is_some() {
      if inp.name.as_ref().unwrap() == input_name {
        if inp.type_.value.is_some() {
          match &inp.type_.value.as_ref().unwrap() {
            Value::TensorType(t) => {
              if t.shape.is_some() {
                for el in &t.shape.as_ref().unwrap().dim {
                  if el.value.is_some() {
                    match el.value.as_ref().unwrap() {
                      DimValue(v) => { shape.push(v) }
                      DimParam(_) => {}
                    }
                  }
                }
              } else {
                panic!("ERROR WHILE GETTING SHAPE OF NODE {}", input_name)
              }
            }
            Value::SequenceType(_) => {}
            Value::MapType(_) => {}
            Value::OptionalType(_) => {}
            Value::SparseTensorType(_) => {}
          };
        } else {
          panic!("ERROR WHILE GETTING TYPE_ OF NODE {}", input_name)
        }
        break;
      }
    }
  }
  shape
}

fn already_into_initializer(model_initializers: &Vec<TensorProto>, input_name: &str) -> bool{
  for init in model_initializers {
    if init.name.is_some() {
      if init.name.as_ref().unwrap() == input_name {
        return true;
      }
    }
  }
  false
}