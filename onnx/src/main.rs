/*use protobuf::parse_from_bytes;
use protobuf::Message;
use std::str;*/

use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Default, Debug)]
struct Value {
    pub optional: String,
    pub attribute_name: String,
    pub value_type: String,
    pub tag: i32
}

impl Value {
    fn new() -> Self {
        Self {
            optional: Default::default(),
            attribute_name: Default::default(),
            value_type: Default::default(),
            tag: Default::default()
        }
    }
}

#[derive(Debug)]
struct Proto {
    pub name: String,
    pub attributes: Vec<Value>
}

impl Proto {
    fn new() -> Self {
        Self {
            name: String::new(),
            attributes: Vec::new()
        }
    }
}

/*

TODO:

0 -> leggere il proto.onnx 
     generare le struct ModelProto, GraphProto, NodeProto,... seguendo le annotations
     /
     altrimenti ricercare nel file (ad es. ModelProto) e poi da li ricercare il FIELD_NUMBER che serve...

1 -> leggere il .onnx
    //08 -> 0000 1000
    //wire_type = 0, field_number = 1
    //ho gia creato un ModelProto, cerco il field con number = filed_number e creo la struct che mi dice(i.e. Graph)

3 -> riguardo alla correttezza sintattica possiamo seguire il principio per cui:
        se io voglio creare un ModelProto all'interno di un GraphProto, cercherà di creare un'istanza di un oggetto che
        non è disponibile nella struct GraphProto. Daremo lì l'errore.

*/


fn main() -> io::Result<()> {
    // Leggi il contenuto del file ONNX in un buffer
    //let onnx_bytes = std::fs::read("models/squeezenet1.0-8.onnx").expect("Failed to read file");
    //println!("{:?}", onnx_bytes);

    let file_path = "../onnx.proto";

    // Apri il file in lettura
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Crea una variabile per tenere traccia del numero di riga
    let mut line_number = 1;

    let mut find_struct = false;
    let mut structures: Vec<Proto> = Vec::new();
    let mut position = 0;

    // Scorri ogni riga del file
    for line in reader.lines() {
        let line = line?;

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
                                println!("ERRORE TAG A LINEA: {}", line_number)
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

    println!("{:?}", structures);

    Ok(())
}
