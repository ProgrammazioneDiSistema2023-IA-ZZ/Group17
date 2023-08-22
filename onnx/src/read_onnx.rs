use std::collections::HashMap;
use std::num::ParseIntError;
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
    let mut number_of_concatenated_bytes: i32 = 0;

    let mut value: i32 = 0;
    let mut length_object_or_enum_field_numer: i32;

    let mut lifo_stack_length: Vec<i32> = Vec::new();
    lifo_stack_length.push(onnx_bytes.len() as i32);
    let mut lifo_stack_struct: Vec<String> = Vec::new();
    lifo_stack_struct.push("modelproto".to_string());

    while counter < onnx_bytes.len() {
        /* Byte[10] to binary */
        let mut binary_string = format!("{:b}", onnx_bytes[counter]);
        number_of_concatenated_bytes = 0;

        /* It means that the binary number starts with bit 1. Dependant information contained between the following bytes */
        if binary_string.len() >= 8 {
            binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
        }

        /* Siccome le stringhe binarie hanno lunghezze diverse (tra 0 e 7) serve sapere a che posizione si trovano gli ultimi 3 bit */
        // PER I BYTE CORRELATI, IL PROBLEMA NON SUSSISTE PERCHE VENGONO AGGIUNTI IN TESTA GLI 0 MANCANTI
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

        if field_name == "raw_data" {
            println!("CIAO");
        }

        if !is_simple_type(&field_type) {
            decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, 2 + number_of_concatenated_bytes); /* Uno per il WT + FN e l'altro per la length*/
            counter += 1;

            /* Byte to binary */
            let mut length_binary_or_enum_filed_number = format!("{:b}", onnx_bytes[counter]);

            if length_binary_or_enum_filed_number.len() >= 8 {
                length_binary_or_enum_filed_number = concat_bytes(length_binary_or_enum_filed_number, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes);
            }

            length_object_or_enum_field_numer = u64::from_str_radix(&*length_binary_or_enum_filed_number, 2).unwrap() as i32;
            let is_enum_with_type = search_enum_in_proto_structure(proto_structure, &field_type, length_object_or_enum_field_numer);

            if is_enum_with_type.is_empty() {
                lifo_stack_struct.push(field_type.clone());
                lifo_stack_length.push(length_object_or_enum_field_numer);
                println!("({}) In {}/{} -> {}, {} ({})", field_number, lifo_stack_struct.get(lifo_stack_struct.len() - 2).unwrap(), lifo_stack_struct.last().unwrap(), field_name, length_object_or_enum_field_numer, wire_type);
            } else {
                println!("({}) In {}/{} -> {} = {} ({})", field_number, lifo_stack_struct.get(lifo_stack_struct.len() - 2).unwrap(), lifo_stack_struct.last().unwrap(), field_name, is_enum_with_type, wire_type);
            }
        } else if wire_type == "LEN" {
            counter += 1;
            /* Byte to binary */
            binary_string = format!("{:b}", onnx_bytes[counter]);
            if binary_string.len() >= 8 {
                binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
            }
            value = u64::from_str_radix(&*binary_string, 2).unwrap() as i32;

            let mut string_result = String::new();
            for i in 1..=value {
                match binary_string_to_ascii(format!("{:b}", onnx_bytes[counter + i as usize])) {
                    Some(ascii_char) => string_result.push(ascii_char),
                    None => println!("Conversione fallita."),
                }
            }

            println!("({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, string_result, wire_type);

            counter += value as usize;
            decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, value + 2 + number_of_concatenated_bytes); /* Uno per WT + FN, value per la lunghezza della stringa e 1 per il campo dimensione della stringa */
        } else {
            if field_type == "float" {
                let mut concat_part: String = String::new();

                counter += 4;

                for n_byte in 0..4 {
                    let formatted = format!("{:02X}", onnx_bytes[counter - n_byte]);
                    concat_part = format!("{}{}", concat_part, formatted);
                }

                // Converti la rappresentazione esadecimale in un intero a 32 bit
                let int_value = u32::from_str_radix(&*concat_part, 16).unwrap();

                // Trasforma l'intero in un array di byte
                let bytes: [u8; 4] = int_value.to_le_bytes();

                // Utilizza il transmute per convertire l'array di byte in un numero float
                let float_value: f32 = unsafe { std::mem::transmute(bytes) };

                println!("({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, float_value, field_type);

                decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, 5 + number_of_concatenated_bytes); //Il numero float è rappresentato su 4 byte
            } else {
                counter += 1;
                /* Byte to binary */
                /* It means that the binary number starts with bit 1. Dependant information contained between the following bytes */
                binary_string = format!("{:b}", onnx_bytes[counter]);
                if binary_string.len() >= 8 {
                    binary_string = concat_bytes(binary_string, &mut counter, &onnx_bytes, &mut number_of_concatenated_bytes)
                }
                value = u64::from_str_radix(&*binary_string, 2).unwrap() as i32;

                println!("({}) In {} => {} = {} ({})", field_number, lifo_stack_struct.last().unwrap(), field_name, value, wire_type);

                decrement_length(&mut lifo_stack_length, &mut lifo_stack_struct, 2 + number_of_concatenated_bytes);
            }
        }

        counter += 1;
    }
}

fn concat_bytes(start_string: String, counter: &mut usize, onnx_bytes: &Vec<u8>, number_bytes: &mut i32) -> String {
    let mut count_parts = 0;
    let mut concat_part: String = String::new();

    let mut binary_string = start_string.clone();
    //following byte is related to previous byte
    while binary_string.len() >= 8 {
        count_parts += 1;
        *counter += 1;
        *number_bytes += 1;
        binary_string = format!("{:b}", onnx_bytes[*counter]);
    }
    count_parts += 1;

    /*the following code drops the MSB, concatenates in Little Endian the bytes and drops the exceeding msb zeros*/
    for i in 0..count_parts {
        let a = *counter + 1 - (count_parts - i);
        let mut part = format!("{:b}", onnx_bytes[*counter + 1 - (count_parts - i)]);
        //drops of the first bit which value is 1 (except for the last byte to concatenate, which value is 0)
        if i < count_parts - 1 {
            part = format!("{}", &part[1..]);
        }

        //little endian concatenation of the inner bytes and drop of the msb 0s
        if i != 0 {
            concat_part = format!("{}{}", part, concat_part);
            concat_part = format!("{:b}", u32::from_str_radix(&*concat_part, 2).unwrap());
        } else {
            concat_part = part;
        }
        binary_string = concat_part.clone();
    }
    if binary_string.len() % 8 != 0 { //if the resulting bytes are not multiple of 8 (8bit=1byte), then padding 0s are added at the head of the string
        let mut padding_zeros = (binary_string.len() as f64 / 8f64).ceil(); //round to upper integer
        padding_zeros = ((padding_zeros * 8.0) - binary_string.len() as f64);
        for _j in 0..padding_zeros as usize {
            binary_string = format!("0{}", binary_string);
        }
    }

    binary_string
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
                            KindOf::Enum => continue,
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
        _ => "not found".to_string()
    }
}


fn search_enum_in_proto_structure(map: &HashMap<String, Proto>, enum_name: &String, tag_value: i32) -> String {
    if map.is_empty() {
        return String::default();
    }
    return match map.get(enum_name) {
        Some(proto) => {
            match proto.kind_of {
                KindOf::Enum => {
                    return String::from(&proto.attributes.get(&tag_value).unwrap().attribute_type);
                }
                _ => String::default()
            }
        }
        None => {
            let mut ret_value = String::default();
            for (_el_name, el_content) in map {
                ret_value.push_str(&search_enum_in_proto_structure(&el_content.contents, enum_name, tag_value));
            }
            return ret_value;
        }
    };
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