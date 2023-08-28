
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

pub mod onnx_structure;

mod read_proto;
use read_proto::create_struct_from_proto_file;

mod read_onnx;
use read_onnx::read_onnx_file;

fn main() {
  let proto_path = "../onnx.proto";
  let onnx_path = "models/model.onnx";
  match create_struct_from_proto_file(proto_path) {
    Ok(result) => {
      println!("{:?}", result);
      read_onnx_file(onnx_path, &result);
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