pub(crate) mod proto_structure;

use std::collections::{BTreeMap, HashMap};
use proto_structure::*;

use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use crate::read_proto::proto_structure::ProtoAttribute;

pub fn create_struct_from_proto_file(file: &str) -> Result<BTreeMap<String, BTreeMap<i32, ProtoAttribute>>, String> {
  //opening file in read mode
  let file = File::open(file).expect("Failed to open and read file .proto");
  let reader = BufReader::new(file);

  let mut proto_map: BTreeMap<String, BTreeMap<i32, ProtoAttribute>> = BTreeMap::new(); //this data structure maintains all the proto structures contained in the .proto file (i.e. Message)
  //let mut proto_map_index = 0; //this index points the message structure which is being managed
  let mut current_proto_name = String::new();
  //TODO: capire se si vuol dare compatibilità anche con la versione 3 di proto. Questa è più complicata da leggere perché gli attributi non iniziano con un annotazione...
  let mut _proto_version = 2; //since version 2 and 3 are valid, this application needs to work properly with both versions

  for cur_line in reader.lines() {
    let mut line = cur_line.expect("Failed to read line from .proto file");
    line = line.to_lowercase().trim_start().parse().unwrap();

    //skipping commented lines
    if line.starts_with("//") || line.starts_with("enum") || line.starts_with("}") ||line == "" {
      continue;
    }

    println!("{:?}", line.to_lowercase());
    let val = line.to_lowercase().starts_with("//");
    println!("{}", val);


    //line starts with "syntax" word (not case sensitive)
    if line.starts_with("syntax"){
      //i.e. syntax = "proto3"; the 3 needs to be extracted and saved
      let trimmed_string = line.trim();
      let mut words = trimmed_string.split_whitespace();
      if let Some(version) = words.nth(2) {
        _proto_version = (&version[6..7]).parse().unwrap();
      }else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }
      continue;
    }

    //line starts with "message" word (not case sensitive)
    if line.starts_with("message"){
      let trimmed_string = line.trim();

      //since the "message" word was found, its following value must be saved:
      // i.e. message Person -> "Person" is what it's needed to be saved (it is represented by word at first position, once the line has been trimmed)
      let mut words = trimmed_string.split_whitespace();
      if let Some(name) = words.nth(1) {
        //proto structure creation and allocation into "proto_map" hashmap
        //let mut p = Proto::new();
        //p.name = name.to_string();
        //proto_map.push(p);
        current_proto_name = name.to_string();

        if current_proto_name.to_lowercase() == "nodeproto" {
          println!("okok");
          let asd = 123;
        }

        match proto_map.get(&current_proto_name) {
          Some(_) => { return Err("Cannot insert duplicated values into hashmap.".to_string()); },
          None => proto_map.insert(name.to_string(), BTreeMap::new())
        };
        //proto_map_index+=1;
      } else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }
      continue;
    }

    if line == "repeated typeproto type_protos = 15;// list of type protos"{
      let asd = 123;
    }

    //current line contains an attribute of a message structure
    if line.contains("optional") || line.contains("repeated") || line.contains("required") || line.contains("map"){
      let mut words = line.split_whitespace();

      if let Some(annotation) = words.next() {
        if let Some(attribute_type) = words.next() {
          if let Some(attribute_name_with_equals) = words.next() {
            let attribute_name: &str = attribute_name_with_equals.split('=').collect::<Vec<&str>>()[0];
            words.next();
            if let Some(tag) = words.next() {
              match proto_map.get_mut(&current_proto_name) {
                Some(proto_attribute_map) => {
                  let mut attribute = ProtoAttribute::new();
                  attribute.annotation = annotation.parse().unwrap();
                  attribute.attribute_type = attribute_type.parse().unwrap();
                  attribute.attribute_name = attribute_name.parse().unwrap();
                  if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {
                    match proto_attribute_map.get(&tag) {
                      Some(_) => { return Err("Cannot insert duplicated values into BTreeMap.".to_string()); },
                      None => proto_attribute_map.insert(tag, attribute)
                    };
                  } else {
                    return Err("Cannot get TAG from .proto file".to_string());
                  }
                },
                None => {return Err("Cannot insert duplicated values into BTreeMap.".to_string());}
              };
              //let mut attribute = ProtoAttribute::new();
              //attribute.annotation = annotation.parse().unwrap();
              //attribute.attribute_type = attribute_type.parse().unwrap();
              //attribute.attribute_name = attribute_name.parse().unwrap();
              //if let Ok(tag) = tag.trim_end_matches(";").parse::<i32>() {
              //  attribute.tag = tag;
              //} else {
              //  return Err("Cannot get TAG from .proto file".to_string());
              //}

              //proto_map.get_mut(proto_map_index - 1).unwrap().attributes.push(attribute);
            }
          }
        }
      }
    }
  }

  Ok(proto_map)
}
