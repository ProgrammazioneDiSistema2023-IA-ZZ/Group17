pub(crate) mod proto_structure;
use proto_structure::Value;
use proto_structure::Proto;

use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn create_struct_from_proto_file(file: String) -> Result<Vec<Proto>, String> {
    // Apri il file in lettura
    let file = File::open(file).expect("Failed to read file");
    let reader = BufReader::new(file);

    // Crea una variabile per tenere traccia del numero di riga
    let mut line_number = 1;

    let mut structures: Vec<Proto> = Vec::new();
    let mut position = 0;

    // Scorri ogni riga del file
    for line in reader.lines() {
        let line = line.expect("Failed to read line");

        // Cerca la parola "message" all'interno della riga (ignorando maiuscole e minuscole)
        if line.to_lowercase().contains("message") && !line.to_lowercase().contains("//") {
            let trimmed_string = line.trim();

            // Dividi la stringa in parole (separate da spazi bianchi) e prendi la seconda parola
            let mut words = trimmed_string.split_whitespace();
            if let Some(word) = words.nth(1) {
                position += 1;

                let mut p = Proto::new();
                p.name = word.to_string();
                structures.push(p);
            } else {
                println!("ERROR!");
            }
        }
        //Cerco gli attributi della struct
        if (line.to_lowercase().contains("optional") || line.to_lowercase().contains("repeated")) && !line.to_lowercase().contains("//") {
            let mut words = line.split_whitespace();

            // Estrai i tre pezzi di informazione
            if let Some(optional) = words.next() {
                if let Some(data_type) = words.next() {
                    if let Some(attr_name_with_equals) = words.next() {
                        // Rimuovi il carattere '=' dalla stringa dell'attributo
                        let attr_name = attr_name_with_equals.trim_end_matches('=');

                        words.next();

                        if let Some(value) = words.next() {
                            let mut v = Value::new();
                            v.optional = optional.parse().unwrap();
                            v.attribute_name = attr_name.parse().unwrap();
                            v.value_type = data_type.parse().unwrap();
                            if let Ok(trimmed) = value.trim_end_matches(";").parse::<i32>() {
                                v.tag = trimmed;
                            } else {
                                println!("ERRORE TAG A LINEA: {}", line_number);
                                return Err("ERRORE".to_string());
                            }

                            structures.get_mut(position - 1).unwrap().attributes.push(v);
                        }
                    }
                }
            }
        }

        // Incrementa il numero di riga
        line_number += 1;
    }

    Ok(structures)
}