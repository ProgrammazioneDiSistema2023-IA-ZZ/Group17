/*
This structure enumerates the possible kind of annotations an attribute of a Proto message could assume
accordingly to Protocol Buffers v2(proto2) documentation (https://protobuf.dev/programming-guides/proto2/).
  - Optional: means that the attribute could be not present in the Attribute structure assuming its default value.
              Note: in proto3 each attribute without explicit annotation its considered as marked optional by default.
  - Repeated: means that the attribute could be present [0..N] times
  - Required: means that the message struct cannot be considered well-formed if this attribute is not present;
             currently this annotation is no more used but is maintained for backward compatibility
  - Map: means that a certain scalar value has been encoded as "packed" (this is done by default in proto3, while must be specified
         in proto2). e.g. Map<string, i32> shows an i32 value which is packed as a string encoding (with a certain LEN).
 */
use core::fmt::Debug;
use std::collections::HashMap;
use std::str::FromStr;

#[repr(C)]
#[derive(Default, Debug, PartialEq)]
pub enum ProtoAnnotation{
  #[default]
  Optional,
  Repeated,
  Required,
  Map
}
#[derive(Debug, PartialEq, Eq)]
pub struct ParseProtoAnnotationError;

impl FromStr for ProtoAnnotation {
  type Err = ParseProtoAnnotationError;
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "optional" => Ok(ProtoAnnotation::Optional),
      "repeated" => Ok(ProtoAnnotation::Repeated),
      "required" => Ok(ProtoAnnotation::Required),
      "map" => Ok(ProtoAnnotation::Map),
      _ => Err(ParseProtoAnnotationError)
    }
  }
}

/*
This structure contains an Attribute of a Message struct in a .proto file. (e.g. optional string name = 1;)
  - annotation: this annotation specifies a modifier for the attribute. This is only present in proto3 version. (i.e. optional)
  - attribute_name: the name of the attribute (i.e. name)
  - attribute_type: the type of the attribute (i.e. string)
  - tag: this is the number which identifies the attribute (i.e. 1)
 */
#[repr(C)]
#[derive(Default, Debug)]
pub struct ProtoAttribute {
  pub annotation: ProtoAnnotation,
  pub attribute_name: String,
  pub attribute_type: String
}
impl ProtoAttribute {
  pub(crate) fn new() -> Self {
    Self {
      annotation: Default::default(),
      attribute_name: Default::default(),
      attribute_type: Default::default()
    }
  }
}

/*
This structure contains a "message" structure of a .proto file
  - name: represents the name of the structure (e.g. message Person -> name=Person)
  - attributes: this vector contains the list of attributes. Each attribute is represented by an Attribute structure
*/
#[repr(C)]
#[derive(Default)]
pub struct Message<'a> {
  pub attributes: HashMap<i32, ProtoAttribute>, //<tag, ProtoAttribute>
  pub contents: HashMap<String, &'a dyn Proto<'a>> //a message could contain itself others messages/one-of
}
impl<'a> Proto<'a> for Message<'a> {
  fn new() -> Self {
    Self {
      attributes: HashMap::new(),
      contents: HashMap::new()
    }
  }
  fn get_attributes(&self) -> Option<&HashMap<i32, ProtoAttribute>> {
    Some(&self.attributes)
  }

  fn set_attributes(&mut self, key: i32, value: ProtoAttribute) {
    self.attributes.insert(key, value);
  }

  fn get_contents_mut(&'a mut self) -> &'a mut HashMap<String, &'a dyn Proto<'a>>{
    &mut self.contents
  }

  fn set_contents(&'a mut self, key: String, value: &'a dyn Proto<'a>) {
    self.contents.insert(key, value);
  }
}

#[repr(C)]
#[derive(Default, Debug)]
pub struct OneOf {
  pub attributes: HashMap<i32, ProtoAttribute>, //<tag, (annotation::Default, type, name)>
}
impl Proto<'_> for OneOf {
  fn new() -> Self {
    Self {
      attributes: HashMap::new(),
    }
  }
  fn get_attributes(&self) -> Option<&HashMap<i32, ProtoAttribute>> {
    Some(&self.attributes)
  }

  fn set_attributes(&mut self, key: i32, value: ProtoAttribute) {
    self.attributes.insert(key, value);
  }

  fn get_contents_mut(&mut self) -> &mut HashMap<String, &dyn Proto> {
    unimplemented!()
  }

  fn set_contents(&mut self, key: String, value: &'_ dyn Proto) {
    unimplemented!()
  }
}

pub trait Proto<'a>{
  fn new() -> Self where Self: Sized;
  fn get_attributes(&self) -> Option<&HashMap<i32, ProtoAttribute>>;
  fn set_attributes(&mut self, key: i32, value: ProtoAttribute);
  fn get_contents_mut(&'a mut self) -> &'a mut HashMap<String, &'a dyn Proto<'a>>;
  fn set_contents(&'a mut self, key: String, value: &'a dyn Proto<'a>);
}
impl Debug for dyn Proto<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{:?}", self)
  }
}


