enum AttributeType {
    Undefined = 0,
    Float = 1,
    Int = 2,
    String = 3,
    Tensor = 4,
    Graph = 5,
    SparseTensor = 11,
    TypeProto = 13,

    Floats = 6,
    Ints = 7,
    Strings = 8,
    Tensors = 9,
    Graphs = 10,
    SparseTensors = 12,
    TypeProtos = 14
}

struct AttributeProto {
    name: Option<String>,
    ref_attr_name: Option<String>,
    doc_string: Option<String>,
    attribute_type: Option<AttributeType>,
    f: Option<f32>,
    i: Option<i64>,
    s: Option<char>,
    t: Option<TensorProto>,
    g: Option<GraphProto>,
    sparse_tensor: Option<SparseTensorProto>,
    tp: Option<TypeProto>,
    floats: Vec<f32>,
    ints: Vec<i32>,
    strings: Vec<char>,
    tensors: Vec<TensorProto>,
    graphs: Vec<GraphProto>,
    sparse_tensors: Vec<SparseTensorProto>,
    type_protos: Vec<TypeProto>
}

struct ValueInfoProto {
    name: Option<String>,
    value_info_type: Option<TypeProto>,
    doc_string: Option<String>
}

struct NodeProto {
    input: Vec<String>,
    output: Vec<String>,
    name: Option<String>,
    op_type: Option<String>,
    domain: Option<String>,
    attribute: Vec<AttributeProto>,
    doc_string: Option<String>
}

struct TrainingInfoProto {
    initialization: Option<GraphProto>,
    algorithm: Option<GraphProto>,
    initialization_binding: Vec<StringStringEntryProto>,
    update_binding: Vec<StringStringEntryProto>
}

pub struct ModelProto {  //ENTRY POINT
    ir_version: Option<i64>,
    opset_import: Vec<OperatorSetIdProto>,
    producer_name: Option<String>,
    producer_version: Option<String>,
    domain: Option<String>,
    model_version: Option<i64>,
    doc_string: Option<String>,
    graph: Option<GraphProto>,
    metadata_props: Vec<StringStringEntryProto>,
    training_info: Vec<TrainingInfoProto>,
    functions: Vec<FunctionProto>
}

impl ModelProto {
    pub fn new() -> Self {
        Self {
            ir_version: Default::default(),
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
    }
}

struct StringStringEntryProto {
    key: Option<String>,
    value: Option<String>
}

struct TensorAnnotation {
    tensor_name: Option<String>,
    quant_parameter_tensor_names: Vec<StringStringEntryProto>
}

struct GraphProto {
    node: Vec<NodeProto>,
    name: Option<String>,
    initializer: Vec<TensorProto>,
    sparse_initializer: Vec<SparseTensorProto>,
    doc_string: Option<String>,
    input: Vec<ValueInfoProto>,
    output: Vec<ValueInfoProto>,
    value_info: Vec<ValueInfoProto>,
    quantization_annotation: Vec<TensorAnnotation>
}

enum DataType {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20
}

struct TensorProto {
    dims: Vec<i64>,
    data_type: Option<i32>,
    segment: Option<Segment>,
    float_data: Vec<f32>,
    int32_data: Vec<i32>,
    string_data: Vec<char>,
    int64_data: Vec<i64>,
    name: Option<String>,
    doc_string: Option<String>,
    raw_data: Option<i8>,
    external_data: Vec<StringStringEntryProto>,
    data_location: Option<DataLocation>,
    double_data: Vec<f64>,
    uint64_data: Vec<u64>
}

struct Segment {
    begin: Option<i64>,
    end: Option<i64>
}

enum  DataLocation {
    DEFAULT = 0,
    EXTERNAL = 1
}

struct SparseTensorProto {
    values: Option<TensorProto>,
    indices: Option<TensorProto>,
    dims: Vec<i64>
}

struct TensorShapeProto {
    dim: Vec<Dimension>
}

enum ValueDimension {
    DimValue(i64),
    DimParam(String)
}

struct Dimension {
    value: ValueDimension,
    denotation: Option<String>
}

struct TypeProto {
    value: Box<ValueTypeProto>,
    denotation: Option<String>
}

struct Tensor {
    elem_type: Option<i32>,
    shape: Option<TensorShapeProto>
}

struct Sequence {
    elem_type: Option<Box<TypeProto>>
}

struct Map {
    key_type: Option<i32>,
    value_type: Option<TypeProto>
}

struct Optional {
    elem_type: Option<TypeProto>
}

struct SparseTensor {
    elem_type: Option<i32>,
    shape: Option<TensorShapeProto>
}

struct Opaque {
    domain: Option<String>,
    name: Option<String>
}

enum ValueTypeProto {
    TensorType(Tensor),
    SequenceType(Box<Sequence>),
    MapType(Map),
    OptionalType(Optional),
    SparseTensorType(SparseTensor),
    OpaqueType(Opaque)
}

struct OperatorSetIdProto {
    domain: Option<String>,
    version: Option<i64>
}

enum OperatorStatus {
    EXPERIMENTAL = 0,
    STABLE = 1
}

struct FunctionProto {
    name: Option<String>,
    input: Vec<String>,
    output: Vec<String>,
    attribute: Vec<String>,
    attribute_proto: Vec<AttributeProto>,
    node: Vec<NodeProto>,
    doc_string: Option<String>,
    opset_import: Vec<OperatorSetIdProto>,
    domain: Option<String>
}