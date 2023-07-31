
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

mod read_proto;
pub mod onnx_structure;

use read_proto::create_struct_from_proto_file;
use read_proto::proto_structure::*;
use onnx_structure::ModelProto;

fn main() {
  // Leggi il contenuto del file ONNX in un buffer
  //let onnx_bytes = std::fs::read("models/squeezenet1.0-8.onnx").expect("Failed to read file");
  //println!("{:?}", onnx_bytes);

  let file_path = "../onnx.proto";
  match create_struct_from_proto_file(file_path) {
    Ok(result) => {
      println!("{:?}", result);
      //read_onnx(&result);
    },
    Err(err) => {
      println!("{}", err);
    }
  }
}

/*
fn read_onnx(proto_structure: &Vec<Proto>) {
  let onnx_bytes = std::fs::read("models/model.onnx").expect("Failed to read file");
  let mut counter = 0;

  let _model_proto = ModelProto::new();
  let current_struct = "ModelProto".to_string();

  while counter < onnx_bytes.len() {
    let binary_string = format!("{:b}", onnx_bytes[counter]);
    if binary_string.len() < 8 {
      //Significa che il primo bit non può essere 1, quindi è un informazione a se
      // Otteniamo l'indice di partizione per dividere la stringa
      let partition_index = binary_string.len().saturating_sub(3);

      // Dividiamo la stringa in due parti: tutto tranne le ultime tre cifre e le ultime tre cifre
      let (first_part, last_three_digits) = binary_string.split_at(partition_index);

      let _wire_type = get_wire_type(last_three_digits);
      let field_number =  u64::from_str_radix(first_part, 2).unwrap();

      let field_name = get_field(&current_struct, field_number, proto_structure).unwrap();

      println!("{}", field_name);
    } else {
      //Il byte dopo è parte dell'informazione
    }

    counter += 1;
  }
}

fn get_field<'a>(current_struct: &String, field_number: u64, proto_structure: &Vec<Proto>) -> Option<String> {
  for el in proto_structure {
    if el.name == current_struct.to_string() {
      for at in &el.attributes {
        //if at.tag == field_number as i32 {
          return Some(at.attribute_name.clone());
        //}
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

 */
