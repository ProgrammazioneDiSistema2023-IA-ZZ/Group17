use std::collections::HashMap;
use std::{io, thread};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use ndarray::{Array1, Array2, Array3, Array4, Axis, concatenate};
use num_traits::Float;
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


/*
This function make the inference on the model received in input
  -It takes 3 parameters:
    ~ model: struct that contains the onnx model
    ~ input_data: this is the input vector of the model (i.e image of a cat)
    ~ input_tensor_name: name(s) of the model's input(s)
  -It prints intermediate type of operations, threads that are working and final result.
*/
pub fn inference(model: ModelProto, input_data: Vec<f32>, input_tensor_name: Vec<&str>) {
  let hashmap_outputs_to_inputs: Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>> = Arc::new(Mutex::new(HashMap::new()));
  let arc_model= Arc::new(model);

  /* Used by main thread while the node considered hasn't already ready inputs data (they will be generated by other threads) */
  let condition_var: Arc<(Mutex<Vec<String>>, Condvar)> = Arc::new((Mutex::new(Vec::new()), Condvar::new()));

  let mut position = 0;
  /* Positions of nodes that has already executed by threads. Main has to skip this nodes */
  let mut position_to_skip: Vec<i32> = Vec::new();

  let mut found_indipendent_nodes = false;

  manage_input_data(&hashmap_outputs_to_inputs, &arc_model, input_data, input_tensor_name);

  let map = hashmap_outputs_to_inputs.lock().unwrap();
  let result = search_node_without_previous_dependencies(&arc_model, map.keys().collect());
  drop(map);  /* For unlocking the lock. If not, main thread could not try to execute nodes while threads works */

  let mut threads: Vec<io::Result<JoinHandle<()>>> = Vec::new();

  if result.is_some() {
    /* Launching threads */
    found_indipendent_nodes = true;

    position_to_skip = result.clone().unwrap().1;
    let independent_nodes = result.unwrap().0;

    let mut n_t = 0;
    for node in independent_nodes {
      let t_map = hashmap_outputs_to_inputs.clone();
      let t_model = arc_model.clone();
      let t_condvar = condition_var.clone();

      threads.push(thread::Builder::new()
        .name(format!("{}{}", "Thread", n_t))
        .spawn(move || {
          node_inference(&node, &t_map, &t_model);

          /* Notify to main thread that some nodes are executed */
          let (l, cvar) = &*t_condvar;
          let mut new_value_added = l.lock().unwrap();
          new_value_added.extend(node.output.clone());
          cvar.notify_all();
        }));

      n_t += 1;
    }
  }

  for node in &arc_model.graph.node {
    //print!("TRY INFERENCE ON {:?} OVER {} OPERATION by main", node.input);

    if found_indipendent_nodes {
      check_pararrel_nodes_and_start_threads(&arc_model, position, node, &mut position_to_skip, &hashmap_outputs_to_inputs, &condition_var, &mut threads);
      found_indipendent_nodes = false;
    }

    if !position_to_skip.contains(&position) { // Il risultato del nodo non è ancora stato calcolato se è presente in position_to_skip
      possibile_wating_for_previous_results(node, &hashmap_outputs_to_inputs, &condition_var, &arc_model);

      check_pararrel_nodes_and_start_threads(&arc_model, position, node, &mut position_to_skip, &hashmap_outputs_to_inputs, &condition_var, &mut threads);
    }

    position += 1;
  }


  for t in threads {
    t.expect("PROBLEM JOINING").join().expect("ERROR");
  }
}

/*
This function allow the Main to stop if the inputs of the considered nodes aren't present (They will be calculated by others threads).
  -It takes 4 parameters:
    ~ node: the considered node in the model
    ~ hasmpa_outputs_to_inputs: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ condition_var: the variable used for waiting in case of the inputs aren't present
    ~ arc_model: smart pointer that contains the onnx struct
*/
pub fn possibile_wating_for_previous_results(node: &NodeProto, hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, condition_var: &Arc<(Mutex<Vec<String>>, Condvar)>, arc_model: &Arc<ModelProto>) {
  let mut inputs_are_present = false;

  while !inputs_are_present {
    inputs_are_present = true;

    for input in &node.input {
      let map = hashmap_outputs_to_inputs.lock().unwrap();
      if !map.contains_key(input) {
        if !already_into_initializer(&arc_model.graph.initializer, input.as_str()) {
          inputs_are_present = false;
          break;
        }
      }
    }

    if inputs_are_present {
      node_inference(&node, &hashmap_outputs_to_inputs, &arc_model);
    } else {
      println!("MAIN THREAD WAITING FOR CHILDREN THREADS RESULTS");
      let (l, cvar) = &**condition_var;
      let mut new_values_added = l.lock().unwrap();

      while new_values_added.len() == 0 {
        new_values_added = cvar.wait(new_values_added).unwrap();
      }

      //println!("Values obtained {:?}", new_values_added);
      *new_values_added = Vec::new();
    }
  }
}

/*
This function start threads if there are nodes that can be executed in parallel.
  -It takes 7 parameters:
    ~ arc_model: smart pointer that contains the onnx model
    ~ position: position of the node in the onnx model
    ~ node: considered node
    ~ position_to_skip: positions of the nodes that are already been executed (in terms of inference)
    ~ hashmap_outputs_to_inputs: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ condition_var: the variable used for notifying that result(s) is ready
    ~ threads: vector that contains all the generated threads
*/
pub fn check_pararrel_nodes_and_start_threads(arc_model: &Arc<ModelProto>, position: i32, node: &NodeProto, position_to_skip: &mut Vec<i32>, hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, condition_var: &Arc<(Mutex<Vec<String>>, Condvar)>, threads: &mut Vec<io::Result<JoinHandle<()>>>) {
  let result = search_node_who_shares_input(&arc_model.graph.node[position as usize + 1..arc_model.graph.node.len() as usize], &node.output[0]);
  if result.is_some() {
    /* Lanciare thread */
    let mut vec_to_add = result.clone().unwrap().1;
    let parallel_nodes = result.unwrap().0;

    for el in vec_to_add.iter_mut() {
      *el += position as i32 + 1;
    }
    position_to_skip.extend(vec_to_add);

    let mut n_t = 0;
    for group in parallel_nodes {
      let t_map = hashmap_outputs_to_inputs.clone();
      let t_model = arc_model.clone();
      let t_condvar = condition_var.clone();

      threads.push(thread::Builder::new()
        .name(format!("{}{}", "Thread", n_t))
        .spawn(move || {
          for n in  group {
            node_inference(&n, &t_map, &t_model);

            let (l, cvar) = &*t_condvar;
            let mut new_value_added = l.lock().unwrap();
            new_value_added.push(n.output[0].clone());
            cvar.notify_all();
          }
        }));

      n_t += 1;
    }
  }
}

/*
This function execute the inference operation of the node.
  -It takes 3 parameters:
    ~ node: inference node
    ~ hashmap_outputs_to_inputs: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ model: smart pointer that contains the onnx model
*/
pub fn node_inference(node: &NodeProto, hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, model: &Arc<ModelProto>) {
  println!("INFERENCE ON INPUT(s) {:?} OVER {} OPERATION done by {}", node.input, node.op_type.clone().unwrap() ,thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let operation = match &node.op_type {
    None => { panic!("Operation {:?} NOT found for node {}", &node.op_type, &node.name.as_ref().unwrap()) }
    Some(op) => { op }
  };

  match operation.as_str() {
    "Conv" => convolution_op(hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
    "Relu" => relu_op(hashmap_outputs_to_inputs, node),
    "MaxPool" => max_pool_op(hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
    "Concat" => concatenate_op(hashmap_outputs_to_inputs, node),
    "Dropout" => drop_out_op(hashmap_outputs_to_inputs, node),
    "GlobalAveragePool" => global_average_pool_op(hashmap_outputs_to_inputs, node),
    "Softmax" => softmax_op(hashmap_outputs_to_inputs, node),
    "Reshape" => reshape_op(hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
    "Add" => add_op(hashmap_outputs_to_inputs, node, &model.graph.input, &model.graph.initializer),
    "MatMul" => mul_op(hashmap_outputs_to_inputs, node),
    _ => { panic!("INFERENCE OPERATION '{}' NOT FOUND FOR NODE {}", operation.as_str(), &node.name.as_ref().unwrap()) }
  }
}

/*
This function insert into initializers the input(s) data of the onnx model
  -It takes 4 parameters:
    ~ hashmap_outputs_to_inputs: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ model: smart pointer that contains the onnx model
    ~ input_data: model inputs(s)
    ~ input_tensor_names: names of the model inputs
*/
fn manage_input_data(hashmap_outputs_to_inputs: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, model: &Arc<ModelProto>, input_data: Vec<f32>, input_tensor_name: Vec<&str>) {
  for input_name in input_tensor_name {
    if !already_into_initializer(&model.graph.initializer, input_name) {
      let dims: Vec<&i64> = search_input_data_shape(&model.graph.input, input_name);
      let array = Array4::from_shape_vec((*dims[0] as usize, *dims[1] as usize, *dims[2] as usize, *dims[3] as usize), input_data.clone()).unwrap();
      let mut map = hashmap_outputs_to_inputs.lock().unwrap();
      map.insert(input_name.to_string(), (None, Some(array)));
    }
  }
}

/*
This function do the convolution
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which convolution has to be executed
    ~ model_inputs: inputs of the onnx model
    ~ model_initializers: initializers of the onnx model
*/
fn convolution_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) {
  let map = output_container.lock().unwrap();
  let input_image = match map.get(&node.input[0]).clone() {
    Some(image) => image.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };
  let kernel = match map.get(&node.input[1]).clone() {
    Some(ker) => ker.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };

  drop(map); /* Per rilasciare il lock */

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
        "dilations" => dilations = Some(Array2::from_shape_vec((1, 2), vec![attr.ints[0] as i32, attr.ints[1] as i32]).unwrap()),
        "group" => group = Some(attr.i.unwrap() as i32),
        "kernel_shape" => {}
        "pads" => pads = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        "strides" => strides = attr.ints.clone().iter().map(|&x| x as f32).collect::<Vec<f32>>().into(),
        _ => panic!("ATTRIBUTE NAME FOR CONVOLUTION NOT FOUND, {}", attr.name.as_ref().unwrap().as_str())
      }
    }
  }

  //dbg!(input_image.clone());
  //dbg!(kernel.clone());
  if !pads.is_empty() {
    if pads[[0]] > 0.0 || pads[[1]] > 0.0 || pads[[2]] > 0.0 || pads[[3]] > 0.0 {
      auto_pad = PadConv::NotSet;
    }
  }
  //println!("PADS: {:?}, STRIDES: {:?}, DILATIONS: {:?}", pads, strides, dilations);
  let conv_layer = ConvLayerConv::new_onnx_tensor_flow(kernel.clone(), bias, auto_pad, dilations, group, pads, strides);
  let output_layer: Array4<f32> = conv_layer.convolve(&input_image);

  //dbg!("Conv: {:?}", output_layer.clone());
  //println!("Conv: {:?}", output_layer.clone());
  println!("Convolve, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

/*
This function do the relu
  -It takes 2 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which relu has to be executed
*/
fn relu_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let output_layer: Array4<f32> = relu(&input);

  //dbg!("Relu: {:?}", output_layer.clone());

  println!("Relu, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

/*
This function do the maxpool
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which maxpool has to be executed
    ~ model_inputs: inputs of the onnx model
    ~ model_initializers: initializers of the onnx model
*/
fn max_pool_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) {
  let map = output_container.lock().unwrap();

  let input_image = match map.get(&node.input[0]).clone() {
    Some(image) => image.1.clone().unwrap(),
    None => {
      let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
      arr4.unwrap()
    }
  };

  drop(map);

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

  //dbg!("MaxPool: {:?}", output_layer.clone());
  println!("MaxPool, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

/*
This function do the concatenate
  -It takes 2 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which concatenate has to be executed
*/
fn concatenate_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input_1 = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
  let input_2 = Array4::from(map.get(node.input[1].as_str()).unwrap().1.clone().unwrap().clone());

  drop(map);

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
  println!("Concatenate, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

/*
This function do the dropout
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which dropout has to be executed
*/
fn drop_out_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

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

  //dbg!("Dropout: {:?}", output_layer.0.clone());
  println!("Dropout, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer.0)));
}

/*
This function do the global average pool
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which global average pool has to be executed
*/
fn global_average_pool_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let output_layer = global_average_pool(input);

  //dbg!(output_layer);
  println!("GlobalAveragePool, done!");

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (None, Some(output_layer)));
}

/*
This function do the soft max
  -It takes 2 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which softmax has to be executed
*/
fn softmax_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();
  let input = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());

  drop(map);

  let result = softmax(input, None);

  println!("Softmax, done!");
  //dbg!(result.clone());

  let mut i = 0;
  let mut best_class_index = 0;
  let mut best_class_percentage = f32::min_value();
  while i < result.len_of(Axis(1)){
    if result[[0, i]] > best_class_percentage {
      best_class_percentage = result[[0, i]];
      best_class_index = i+1;
    }
    i += 1;
  }
  println!("\nSqueezenet1.0-8 Inference results: Class {}-nth predicted.\nActual Data: {:?}", best_class_index, result.clone());
}

/*
This function do the reshape
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which reshape has to be executed
    ~ model_inputs: inputs of the onnx model
    ~ model_initializers: initializers of the onnx model
*/
#[allow(unused_assignments)]
#[allow(unused_variables)]
fn reshape_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) {
  let mut data: Array4<f32> = Default::default();
  if already_into_initializer(model_initializers, node.input[0].as_str()) {
    let (arr4, _, _, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
    data = arr4.unwrap();
  } else {
    let map = output_container.lock().unwrap();
    data = Array4::from(map.get(node.input[0].as_str()).unwrap().1.clone().unwrap());
    drop(map);
  }

  let mut shape: Array1<i64> = Default::default();
  if already_into_initializer(model_initializers, node.input[1].as_str()) {
    let (_, _, _, _, arr1) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
    shape = arr1.unwrap();
  } else {
    panic!("Cannot get Shape for Reshape operation");
  }

  let output_layer: Array2<f32> = reshape(data, shape, None);

  //dbg!("Reshape: {:?}", output_layer.clone());
  println!("Reshape, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (Some(output_layer), None));
}

/*
This function do the add
  -It takes 4 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which add has to be executed
    ~ model_inputs: inputs of the onnx model
    ~ model_initializers: initializers of the onnx model
*/
#[allow(unused_assignments)]
fn add_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) {
  let mut input_1_arr_4: Array4<f32> = Default::default();
  let mut input_1_arr_2: Array2<f32> = Default::default();
  let mut input_2_arr_3: Array3<f32> = Default::default();
  let mut input_2_arr_2: Array2<f32> = Default::default();

  let map = output_container.lock().unwrap();

  if already_into_initializer(model_initializers, node.input[0].as_str()) {
    let (arr4, _, arr2, _, _) = get_stored_tensor_for_convolution(0, node, model_inputs, model_initializers);
    match arr4 {
      Some(arr4) => input_1_arr_4 = arr4,
      None => {
        match arr2 {
          Some(arr2) => input_1_arr_2 = arr2,
          None => panic!("Cannot get input 1 for Add operation from initializers")
        }
      }
    };
  } else {
    match map.get(node.input[0].as_str()).unwrap().0.clone() {
      Some(arr2) => input_1_arr_2 = arr2,
      None => {
        match map.get(node.input[0].as_str()).unwrap().1.clone() {
          Some(arr4) => input_1_arr_4 = arr4,
          None => panic!("Cannot get input 1 for Add operation from hashmap input/output")
        }
      }
    }
  }

  drop(map);

  if already_into_initializer(model_initializers, node.input[1].as_str()) {
    let (_, arr3, arr2, _, _) = get_stored_tensor_for_convolution(1, node, model_inputs, model_initializers);
    match arr3 {
      Some(arr3) => input_2_arr_3 = arr3,
      None => {
        match arr2 {
          Some(arr2) => input_2_arr_2 = arr2,
          None => panic!("Cannot get input 2 for Add operation from initializes")
        }
      }
    }
  } else {
    panic!("Cannot get input 2 for Add operation");
  }

  let mut map_mut = output_container.lock().unwrap();

  let mut output_layer_2: Array2<f32> = Default::default();
  let mut output_layer_4: Array4<f32> = Default::default();
  if input_1_arr_4.len() > 0{
    output_layer_4 = input_1_arr_4+input_2_arr_3;
    //dbg!(output_layer_4.clone());
    println!("Add, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));
    map_mut.insert(node.output[0].clone(), (None, Some(output_layer_4)));
  }else{
    output_layer_2 = input_1_arr_2+input_2_arr_2;
    //dbg!("Add: {:?}", output_layer_2.clone());
    println!("Add, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));
    map_mut.insert(node.output[0].clone(), (Some(output_layer_2.clone()), None));

    let mut i = 0;
    let mut best_class_index = 0;
    let mut best_class_percentage = f32::min_value();
    while i < output_layer_2.clone().len_of(Axis(1)){
      if output_layer_2[[0, i]] > best_class_percentage {
        best_class_percentage = output_layer_2[[0, i]];
        best_class_index = i+1;
      }
      i += 1;
    }
    println!("\nMNist-8 Inference results: Class {}-nth predicted.\nActual Data: {:?}", best_class_index, output_layer_2);

    //output_container.insert(node.output[0].clone(), (Some(output_layer_2), None));
  }
}

/*
This function do the mul
  -It takes 2 parameters:
    ~ output_container: contains the partial results calculated by inferences operations (i.e. convolution, relu)
    ~ node: node on which mul has to be executed
*/
fn mul_op(output_container: &Arc<Mutex<HashMap<String, (Option<Array2<f32>>, Option<Array4<f32>>)>>>, node: &NodeProto) {
  let map = output_container.lock().unwrap();

  let input_1 = Array2::from(map.get(node.input[0].as_str()).unwrap().0.clone().unwrap());
  let input_2 = Array2::from(map.get(node.input[1].as_str()).unwrap().0.clone().unwrap());

  drop(map);

  let output_layer: Array2<f32> = input_1.dot(&input_2);

  //dbg!("MatMul: {:?}", output_layer.clone());
  println!("MatMul, done! by {}", thread::current().name().unwrap_or("PROCESSO PRINCIPALE"));

  let mut map_mut = output_container.lock().unwrap();
  map_mut.insert(node.output[0].clone(), (Some(output_layer), None));
}

/*
This function get the input tensors needed by the convolution
  -It takes 4 parameters:
    ~ i: position of the node in the onnx model
    ~ node: node on which convolution has to be executed
    ~ model_inputs: inputs of the onnx model
    ~ model_initializers: initializers of the onnx model
It returns the input tensors
*/
fn get_stored_tensor_for_convolution(i: usize, node: &NodeProto, model_inputs: &Vec<ValueInfoProto>, model_initializers: &Vec<TensorProto>) -> (Option<Array4<f32>>, Option<Array3<f32>>, Option<Array2<f32>>, Option<Array1<f32>>, Option<Array1<i64>>){
  let shape = search_input_data_shape(model_inputs, &node.input[i]);

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
        } else if init.float_data.len() > 0 {
          for float_value in &init.float_data {
            raw_data.push(*float_value);
          }
        } else if init.int64_data.len() > 0 {
          for int_value in &init.int64_data {
            raw_data_i64.push(*int_value);
          }
        }
      }
    }
  }

  if shape.len() == 4 {
    (Some(Array4::from_shape_vec((*shape[0] as usize, *shape[1] as usize, *shape[2] as usize, *shape[3] as usize), raw_data).unwrap()), None, None, None, None)
  } else if shape.len() == 3 {
    (None, Some(Array3::from_shape_vec((*shape[0] as usize, *shape[1] as usize, *shape[2] as usize), raw_data).unwrap()), None, None, None)
  } else if shape.len() == 2 {
    (None, None, Some(Array2::from_shape_vec((*shape[0] as usize, *shape[1] as usize), raw_data).unwrap()), None, None)
  } else {
    if raw_data.len() > 0 {
      (None, None, None, Some(Array1::from_shape_vec(*shape[0] as usize, raw_data).unwrap()), None)
    }else {
      (None, None, None, None, Some(Array1::from_shape_vec(*shape[0] as usize, raw_data_i64).unwrap()))
    }
  }
}

/*
This function searches convert u8 to f32
  -It takes 1 parameters:
    ~ bytes: number to convert
  -It returns the correspondent f32 number
*/
fn u8_to_f32(bytes: &[u8]) -> f32 {
  let mut arr = [0; 4];
  arr.copy_from_slice(bytes);
  f32::from_le_bytes(arr)
}

/*
This function searches the node's input shape(s) into the onnx model.
  -It takes 2 parameters:
    ~ model_inputs: list of the inputs
    ~ input_name: input to search
  -It returns the list of shapes
*/
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

/*
This function searches if a node's input is already in the model's initializers
  -It takes 2 parameters:
    ~ model_initializers: model's initializers
    ~ input: input to check
  -It returns true if it's present, false otherwise
*/
fn already_into_initializer(model_initializers: &Vec<TensorProto>, input_name: &str) -> bool {
  for init in model_initializers {
    if init.name.is_some() {
      if init.name.as_ref().unwrap() == input_name {
        return true;
      }
    }
  }
  false
}

/*
This function searches if in the model there are node that can be executed in parallel (its shares the same input(s))
  -It takes 2 parameters:
    ~ nodes: slice of the nexts nodes
    ~ input_to_check: name of node's output to check if it's shared among different nodes
  -It returns the independents nodes and their position in the onnx model
*/
fn search_node_who_shares_input(nodes: &[NodeProto], input_to_check: &String) -> Option<(Vec<Vec<NodeProto>>, Vec<i32>)> {
  let mut node_shares_input = 0;
  let mut position = 0;
  let mut hash_shares: HashMap<i32, NodeProto> = HashMap::new();

  for node in nodes {
    if node.input.contains(input_to_check) {
      node_shares_input += 1;
      hash_shares.insert(position, node.clone());
    }

    position += 1;
  }

  if node_shares_input >= 2 {
    let mut pos_to_skip: Vec<i32> = hash_shares.keys().cloned().collect();

    let mut vec_sequence: Vec<Vec<NodeProto>> = Vec::new();

    for el in &hash_shares {
      vec_sequence.push(Vec::new());
      vec_sequence.last_mut().unwrap().push(el.1.clone());
      let output_to_find = &el.1.output[0];

      let counter = el.0 + 1;

      for node in &nodes[*el.0 as usize + 1..nodes.len() as usize] {
        if node.input.contains(output_to_find) {
          vec_sequence.last_mut().unwrap().push(node.clone());
          pos_to_skip.push(counter);
        } else {
          break;
          /* Sequence interrupted */
        }
      }
    }

    Some((vec_sequence, pos_to_skip))
  } else {
    None
  }
}

/*
This function searches if in the model there are node without previous dependencies (they can be executed in a separated threads)
  -It takes 2 parameters:
    ~ model: smart pointer that contains the onnx model
    ~ previous_outputs: vector that contains names of the operations that are already been done
  -It returns the independents nodes and their position in the onnx model
*/
fn search_node_without_previous_dependencies(model: &Arc<ModelProto>, previous_outputs: Vec<&String>) -> Option<(Vec<NodeProto>, Vec<i32>)> {
  let mut position = 0;
  let mut pos_to_skip: Vec<i32> = Vec::new();
  let mut indipendent_nodes: Vec<NodeProto> = Vec::new();

  for node in &model.graph.node {
    // let is_contained = node.inputs.inter().all(|&item| previous_outputs.contains(item));
    let mut is_contained = true;
    for input in &node.input {
      if !previous_outputs.contains(&input) {
        if !already_into_initializer(&model.graph.initializer, input.as_str()) {
          is_contained = false;
          break;
        }
      }
    }

    if is_contained {
      pos_to_skip.push(position);
      indipendent_nodes.push(node.clone());
    }

    position += 1;
  }

  if pos_to_skip.len() < 2 {
    None
  } else {
    Some((indipendent_nodes, pos_to_skip))
  }
}