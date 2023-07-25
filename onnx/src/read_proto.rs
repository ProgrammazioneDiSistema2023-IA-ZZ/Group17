pub(crate) mod proto_structure;

use proto_structure::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::read_proto::proto_structure::ProtoAttribute;

pub fn create_struct_from_proto_file(file: &str) -> Result<Vec<Proto>, String> {
  //opening file in read mode
  let file = File::open(file).expect("Failed to open and read file .proto");
  let reader = BufReader::new(file);

  let mut structures: Vec<Proto> = Vec::new(); //this data structure maintains all the structures contained in the .proto file (i.e. Message, Enum structures)
  let mut structures_index = 0; //this index points the message structure which is being managed
  //TODO: capire se si vuol dare compatibilità anche con la versione 3 di proto. Questa è più complicata da leggere perché gli attributi non iniziano con un annotazione...
  let mut _proto_version = 2; //since version 2 and 3 are valid, this application needs to work properly with both versions

  for line in reader.lines() {
    let line = line.expect("Failed to read line from .proto file");

    //skipping commented lines
    if line.to_lowercase().contains("//") {
      continue;
    }

    //line starts with "syntax" word (not case sensitive)
    if line.to_lowercase().contains("syntax"){
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
    if line.to_lowercase().contains("message"){
      let trimmed_string = line.trim();

      //since the "message" word was found, its following value must be saved:
      // i.e. message Person -> "Person" is what it's needed to be saved (it is represented by word at first position, once the line has been trimmed)
      let mut words = trimmed_string.split_whitespace();
      if let Some(name) = words.nth(1) {
        //proto structure creation and allocation into "structures" vector
        let mut p = Proto::new();
        p.name = name.to_string();
        structures.push(p);
        structures_index+=1;
      } else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }
      continue;
    }

    //current line contains an attribute of a message structure
    if line.to_lowercase().contains("optional") || line.to_lowercase().contains("repeated") || line.to_lowercase().contains("required") || line.to_lowercase().contains("map"){
      let mut words = line.split_whitespace();

      if let Some(annotation) = words.next() {
        if let Some(attribute_type) = words.next() {
          if let Some(attribute_name_with_equals) = words.next() {
            let attribute_name = attribute_name_with_equals.trim_end_matches('=');
            words.next();
            if let Some(tag) = words.next() {
              let mut attribute = ProtoAttribute::new();
              attribute.annotation = annotation.parse().unwrap();
              attribute.attribute_type = attribute_type.parse().unwrap();
              attribute.attribute_name = attribute_name.parse().unwrap();
              if let Ok(tag) = tag.trim_end_matches(";").parse::<i32>() {
                attribute.tag = tag;
              } else {
                return Err("Cannot get TAG from .proto file".to_string());
              }

              structures.get_mut(structures_index - 1).unwrap().attributes.push(attribute);
            }
          }
        }
      }
    }
  }

  Ok(structures)
}
