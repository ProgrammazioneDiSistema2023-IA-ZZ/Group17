
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
mod read_onnx;

use read_proto::create_struct_from_proto_file;
use read_proto::proto_structure::*;
use onnx_structure::ModelProto;
use crate::read_onnx::read_onnx_file;

use onnxruntime::environment::Environment;
use onnxruntime::LoggingLevel;
use onnxruntime::GraphOptimizationLevel;

fn main() {
  // Leggi il contenuto del file ONNX in un buffer
  //let onnx_bytes = std::fs::read("models/squeezenet1.0-8.onnx").expect("Failed to read file");
  //println!("{:?}", onnx_bytes);

  let file_path = "../onnx.proto";
  match create_struct_from_proto_file(file_path) {
    Ok(result) => {
      println!("{:?}", result);
      read_onnx_file(&result);
    },
    Err(err) => {
      eprintln!("{}", err);
    }
  }
}

/* NEW VERSION */
/*fn main() -> onnxruntime::Result<()> {
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;

    let session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("models/model.onnx")?;

    println!("{:?}", session);

    Ok(())
}*/