pub(crate) mod proto_structure;

use std::collections::HashMap;
use std::fmt::format;
use proto_structure::*;

use std::fs::File;
use std::hash::Hash;
use std::io::{BufRead, BufReader};
use crate::read_proto::proto_structure::ProtoAttribute;

pub fn create_struct_from_proto_file(file: &str) -> Result<HashMap<String, &'static dyn Proto>, String> {
  //opening file in read mode
  let file = File::open(file).expect("Failed to open and read file .proto");
  let reader = BufReader::new(file);

  let mut proto_map: HashMap<String, &'static dyn Proto> = HashMap::new(); //this data structure maintains all the proto structures contained in the .proto file (i.e. Message)
  //let mut proto_map_index = 0; //this index points the message structure which is being managed
  let mut current_proto_name = String::new();
  //TODO: capire se si vuol dare compatibilità anche con la versione 3 di proto. Questa è più complicata da leggere perché gli attributi non iniziano con un annotazione...
  let mut _proto_version = 2; //since version 2 and 3 are valid, this application needs to work properly with both versions
  let mut message_level = 0;
  let mut parent_type = String::from("message");

  for cur_line in reader.lines() {
    let mut line = cur_line.expect("Failed to read line from .proto file");
    line = line.to_lowercase().trim_start().parse().unwrap();

    //skipping commented lines
    if line.starts_with("//") || line.starts_with("enum") ||  line == "" {
      continue;
    }

    if line.starts_with("}") {
      message_level = 0;
      //println!("was adding into {}", current_proto_name);
      let message_name_path: Vec<&str> = current_proto_name.split('/').collect();
      if message_name_path.len() > 1{
        let mut i = 0;
        let mut aus = String::new();
        while i < message_name_path.len() - 1{
          if i == 0 {
            aus.push_str(message_name_path[i]);
          }else{
            let mut aus_str = String::from("/");
            aus_str.push_str(message_name_path[i]);
            aus.push_str(&aus_str);
          }i+=1;
        }
        current_proto_name = aus;
      }
      //println!("now adding into {}\n", current_proto_name);
    }

    //println!("{:?}", line.to_lowercase());
    //let val = line.to_lowercase().starts_with("//");
    //println!("{}", val);


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
    if line.starts_with("message") || line.starts_with("oneof"){
      message_level+=1;
      if line.starts_with("message"){
        parent_type = String::from("message");
      }else{
        parent_type = String::from("oneof");
      }


      let trimmed_string = line.trim();

      //since the "message" word was found, its following value must be saved:
      // i.e. message Person -> "Person" is what it's needed to be saved (it is represented by word at first position, once the line has been trimmed)
      let mut words = trimmed_string.split_whitespace();
      if let Some(name) = words.nth(1) {
        if message_level == 1 { //the message is at the top level of the hashmap (i.e. message person{})
          current_proto_name = name.to_string();
        }else{ //nested message. Saving the path it way more convenient for searching it than scan all the hashmap nested levels
          // (i.e. message person{ message address {}}) -> person/address
          let mut aus_str = String::from("/");
          aus_str.push_str(name);
          current_proto_name.push_str(&aus_str);
        }

        let message_name_path: Vec<&str> = current_proto_name.split('/').collect();

        if message_name_path.len() > 1 {
          match get_correct_level_hashmap(&mut proto_map, &message_name_path, 0) {
            Ok(map) => {
              match map.get(message_name_path[message_name_path.len() - 1]) {
                Some(_) => { return Err("Cannot insert duplicated values into HashMap.".to_string()); },
                None => {
                  println!("adding {name} into {message_name_path:?}");
                  if line.starts_with("message"){
                    map.insert(name.to_string(), &Message::new());
                  }else{
                    map.insert(name.to_string(), &OneOf::new());
                  }

                }
              };
            },
            Err(err) => { return Err(err); }
          }
        }else {
          match proto_map.get(&current_proto_name) {
            Some(_) => { return Err("Cannot insert duplicated values into HashMap.".to_string()); },
            None => proto_map.insert(name.to_string(), &Message::new())
          };
        }
      } else {
        return Err("Cannot get the specified index of the trimmed line!".to_string());
      }

      continue;
    }

    //current line contains an attribute of a message structure
    if line.starts_with("optional") || line.starts_with("repeated") || line.starts_with("required") || line.starts_with("map"){
      let mut words = line.split_whitespace();

      if let Some(annotation) = words.next() {
        if let Some(attribute_type) = words.next() {
          if let Some(attribute_name_with_equals) = words.next() {
            let attribute_name: &str = attribute_name_with_equals.split('=').collect::<Vec<&str>>()[0];
            words.next();
            if let Some(tag) = words.next() {
              if let Ok(tag) = tag.split(';').collect::<Vec<&str>>()[0].parse::<i32>() {

                let message_name_path: Vec<&str> = current_proto_name.split('/').collect();

                match search_message_in_hashmap(&mut proto_map, &message_name_path, 0){
                  Ok(message) => {
                    match message.get_attributes(){
                      Some(attributes) => {
                        match attributes.get(&tag) {
                          Some(_) => { return Err("Cannot insert duplicated values into HashMap.".to_string()); },
                          None => {
                            let mut attribute = ProtoAttribute::new();
                            attribute.annotation = annotation.parse().unwrap();
                            attribute.attribute_type = attribute_type.parse().unwrap();
                            attribute.attribute_name = attribute_name.parse().unwrap();
                            message.set_attributes(tag, attribute);
                          }
                        };
                      },
                      None => { return Err("Cannot get attributes.".to_string()); }
                    };
                  },
                  Err(err) => {return Err(err);}
                };

              } else {
                return Err("Cannot get TAG from .proto file".to_string());
              }
            }
          }
        }
      }
    }
  }

  Ok(proto_map)
}

fn get_correct_level_hashmap<'a, 'b>(map: &'a mut HashMap<String, &'a dyn Proto>, message_name_path: &'b[&str], index: usize) -> Result<&'a mut HashMap<String, &'a dyn Proto<'a>>, String>{
  return match map.get_mut(message_name_path[index]) {
    Some(message) => {
      if index == message_name_path.len() - 2 {
        //println!("i'm adding into: {:?}", message_name_path[index]);
        /*match ... {
          Some(contents) => Ok(contents),
          None => Err("Cannot get oneof content because it does not exists".to_string())
        }*/
        Ok(message.get_contents_mut())
      } else {
        get_correct_level_hashmap(message.get_contents_mut(), message_name_path, index + 1)

        /*match message.get_contents_mut(){
          Some(contents) => get_correct_level_hashmap(contents, message_name_path, index + 1),
          None => Err("Cannot get oneof content because it does not exists".to_string())
        }*/
      }
    }
    None => { return Err(format!("Cannot get hashmap content. Path: {:?}",message_name_path)); }
  };
}

fn search_message_in_hashmap<'a, 'b>(map: &'a mut HashMap<String, &'a dyn Proto>, message_name_path: &'b[&str], index: usize) -> Result<&'a mut &'a dyn Proto<'a>, String>{
  //println!("searching for: {}", message_name_path[index]);
  //println!("in: {:?}", map);
  return match map.get_mut(message_name_path[index]) {
    Some(message) => {
      if index < message_name_path.len() - 1 {
        search_message_in_hashmap(message.get_contents_mut(), message_name_path, index + 1)
        /*match message.get_contents_mut(){
          Some(contents) => search_message_in_hashmap(contents, message_name_path, index + 1),
          None => Err("Cannot get oneof content because it does not exists".to_string())
        }*/
      } else {
        Ok(message)
      }
    }
    None => { return Err(format!("Cannot find element in hashmap. Path: {:?}",message_name_path)); }
  };
}


