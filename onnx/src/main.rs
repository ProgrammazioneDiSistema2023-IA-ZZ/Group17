use protobuf::parse_from_bytes;
use protobuf::Message;
use std::str;

/*

TODO:
1 -> leggere il .onnx
    //08 -> 0000 1000
    //wire_type = 0, field_number = 1
    //ho gia creato un ModelProto, cerco il field con number = filed_number e creo la struct che mi dice(i.e. GraphProto)

2 -> per ogni message in onnx.proto, va creata la struct corrispondente seguendo le annotations di onnx.proto,
    fornendo la relativa implementazione

3 -> riguardo alla correttezza sintattica possiamo seguire il principio per cui:
        se io voglio creare un ModelProto all'interno di un GraphProto, cercherà di creare un'istanza di un oggetto che
        non è disponibile nella struct GraphProto. Daremo lì l'errore.

*/


fn main() {
    // Leggi il contenuto del file ONNX in un buffer
    //let onnx_bytes = std::fs::read("models/squeezenet1.0-8.onnx").expect("Failed to read file");
    //println!("{:?}", onnx_bytes);
}