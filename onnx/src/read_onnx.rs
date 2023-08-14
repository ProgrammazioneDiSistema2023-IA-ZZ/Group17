use std::collections::HashMap;
use crate::onnx_structure::ModelProto;
use crate::read_proto::proto_structure::{KindOf, Proto, ProtoAttribute};

pub fn read_onnx_file(proto_structure: &HashMap<String, Proto>) {
    let onnx_bytes = std::fs::read("models/squeezenet1.0-8.onnx").expect("Failed to read file");
    let mut counter = 0;

    let model_proto = ModelProto::new();

    let mut wire_type: String = String::new();
    let mut field_number: i32;
    let mut field_name: String = String::new();
    let mut field_type: String = String::new();

    let mut value: i32 = 0;
    let mut length_object: i32;

    let mut lifo_stack_length: Vec<i32> = Vec::new();
    lifo_stack_length.push(onnx_bytes.len() as i32);
    let mut lifo_stack_struct: Vec<String> = Vec::new();
    lifo_stack_struct.push("modelproto".to_string());

    let mut count_parts = 0;
    let mut concat_part: String = String::new();

    while counter < onnx_bytes.len() {
        /* Byte to binary */
        let mut binary_string = format!("{:b}", onnx_bytes[counter]);

        /* It means that the binary number starts with 1. Non indipendent information contained in the number */
        if binary_string.len() >= 8 {
            //Il byte dopo è parte dell'informazione
            while binary_string.len() >= 8 {
                count_parts += 1;
                counter += 1;
                binary_string = format!("{:b}", onnx_bytes[counter]);
            }
            count_parts += 1; /* Per contare l'ultima parte */

            for i in 0..count_parts {
                let mut part = format!("{:b}", onnx_bytes[counter - i]);
                if i != 0 {
                    part = format!("{}{}", '0', &part[1..]); // Crea una nuova stringa con il primo carattere modificato
                    part = format!("{:b}", u32::from_str_radix(&*part, 2).unwrap());
                }

                concat_part = format!("{}{}", concat_part, part);
                binary_string = concat_part.clone();
            }
        }

        /* Siccome le stringhe binarie hanno lunghezze diverse (tra 0 e 7) serve sapere a che posizione si trovano gli ultimi 3 bit */
        let partition_index = binary_string.len().saturating_sub(3);

        /* Get the last three bits */
        let (first_part, last_three_digits) = binary_string.split_at(partition_index);

        wire_type = get_wire_type(last_three_digits);

        field_number = u64::from_str_radix(first_part, 2).unwrap() as i32;

        match get_field(&lifo_stack_struct.last().unwrap(), field_number, proto_structure) {
            Some((f_n, f_t)) => {
                field_name = f_n;
                field_type = f_t;
            }
            None => panic!("ONNX SYNTAX ERROR")
        }

        if !is_simple_type(&field_type) {
            decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, 2); /* Uno per il WT + FN e l'altro per la length*/
            counter += 1;

            lifo_stack_struct.push(field_type.clone());
            /* Byte to binary */
            let length_binary = format!("{:b}", onnx_bytes[counter]);

            length_object = u64::from_str_radix(&*length_binary, 2).unwrap() as i32;
            lifo_stack_length.push(length_object);

            println!("{} -> {}, {} ({})", lifo_stack_struct.last().unwrap(), field_name, length_object, wire_type);
        } else if wire_type == "LEN" {
            counter += 1;
            /* Byte to binary */
            let value_binary = format!("{:b}", onnx_bytes[counter]);
            value = u64::from_str_radix(&*value_binary, 2).unwrap() as i32;

            let mut string_result = String::new();
            for i in 1..=value {
                match binary_string_to_ascii(format!("{:b}", onnx_bytes[counter + i as usize])) {
                    Some(ascii_char) => string_result.push(ascii_char),
                    None => println!("Conversione fallita."),
                }
            }

            println!("In {} => {} = {} ({})", lifo_stack_struct.last().unwrap(), field_name, string_result, wire_type);

            counter += value as usize;
            decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, value + 2); /* Uno per WT + FN, value per la lunghezza della stringa e 1 per il campo dimensione della stringa */
        } else {
            counter += 1;
            /* Byte to binary */
            let value_binary = format!("{:b}", onnx_bytes[counter]);
            value = u64::from_str_radix(&*value_binary, 2).unwrap() as i32;

            println!("In {} => {} = {} ({})", lifo_stack_struct.last().unwrap(), field_name, value, wire_type);

            decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, 2);
        }

        counter += 1;
    }
}

fn decrement_length(vec_length: &mut Vec<i32>, vec_struct: &mut Vec<String>, value_to_decrement: i32) {
    if vec_length.len() > 0 {
        for num in vec_length.iter_mut() {
            *num -= value_to_decrement;
        }

        let mut found_non_zero = false;
        let mut zero_count = 0;

        // Scansiona il vettore al contrario per trovare elementi non zero
        for &num in vec_length.iter().rev() {
            if num != 0 {
                found_non_zero = true;
                break;
            }
            zero_count += 1;
        }

        // Rimuovi gli zeri finali se ce ne sono
        if zero_count > 0 {
            vec_length.truncate(vec_length.len() - zero_count);
            vec_struct.truncate(vec_struct.len() - zero_count);
        }
        /*if let Some(last_element) = vec_length.pop() {
            if last_element == 0 {
                vec_struct.pop();
            } else {
                vec_length.push(last_element);
            }
        } else {
            panic!("UNEXPECTED ERROR!")
        }*/
    }
}

fn is_simple_type(value_type: &String) -> bool {
    ["string", "int64", "float", "bytes", "int32"].iter().any(|&s| s == value_type)
}

fn binary_string_to_ascii(binary_string: String) -> Option<char> {
    if let Ok(binary_num) = u8::from_str_radix(&binary_string, 2) {
        if let Some(ascii_char) = char::from_u32(binary_num as u32) {
            return Some(ascii_char);
        }
    }
    None
}

fn get_field(current_struct: &String, field_number: i32, proto_structure: &HashMap<String, Proto>) -> Option<(String, String)> {
    for el in proto_structure {
        if el.0 == current_struct {
            return match el.1.attributes.get(&field_number) {
                Some(ap) => Some((ap.attribute_name.clone(), ap.attribute_type.clone())),
                None => {
                    /* Cerco se c'è un ONEOF*/
                    let mut found_one_of = false;
                    let mut ret_value = None;
                    for inner_el in &el.1.contents {
                        //println!("{} -> {:?}, {:?}", inner_el.0, inner_el.1.kind_of, inner_el.1.attributes);
                        match inner_el.1.kind_of {
                            KindOf::Message => continue,
                            KindOf::OneOf => {
                                match inner_el.1.attributes.get(&field_number) {
                                    Some(ap) => {
                                        found_one_of = true;
                                        ret_value = Some((ap.attribute_name.clone(), ap.attribute_type.clone()));
                                        break;
                                    }
                                    None => continue
                                }
                            }
                        }
                    }

                    if found_one_of {
                        ret_value
                    } else {
                        None
                    }
                }
            };
        } else if el.1.contents.len() != 0 {
            match get_field(current_struct, field_number, &el.1.contents) {
                None => continue,
                Some(found) => return Some(found)
            }
        }
    }

    None
}

fn get_wire_type(binary_number: &str) -> String {
    // Converti la stringa binaria in un numero u64 in base 2
    let decimal_number = u64::from_str_radix(binary_number, 2).unwrap();

    match decimal_number {
        0 => "VARINT".to_string(),
        1 => "I64".to_string(),
        2 => "LEN".to_string(),
        3 => "SGROUP".to_string(),
        4 => "EGROUP".to_string(),
        5 => "I32".to_string(),
        _ => "NON TROVATO".to_string()
    }
}

/*fn set_attribute_model_proto(attribute_name: String, model_proto: &mut ModelProto, value:) {
    match attribute_name {
        "ir_version".parse().unwrap() => model_proto.
        opset_import: Default::default(),
        producer_name: Default::default(),
        producer_version: Default::default(),
        domain: Default::default(),
        model_version: Default::default(),
        doc_string: Default::default(),
        graph: Default::default(),
        metadata_props: Default::default(),
        training_info: Default::default(),
        functions: Default::default()
    }
}*/
