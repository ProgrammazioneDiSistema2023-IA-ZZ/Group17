#[derive(Default, Debug)]
pub struct Value {
    pub optional: String,
    pub attribute_name: String,
    pub value_type: String,
    pub tag: i32
}

impl Value {
    pub(crate) fn new() -> Self {
        Self {
            optional: Default::default(),
            attribute_name: Default::default(),
            value_type: Default::default(),
            tag: Default::default()
        }
    }
}

#[derive(Debug)]
pub struct Proto {
    pub name: String,
    pub attributes: Vec<Value>
}

impl Proto {
    pub(crate) fn new() -> Self {
        Self {
            name: String::new(),
            attributes: Vec::new()
        }
    }
}