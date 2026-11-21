// The parts of `Result<..>` and view type names.

pub(crate) fn result_parts(name: &str) -> Option<(Type, Type)> {
    fn component(value: &str, depth: usize) -> Option<Type> {
        match value {
            "i32" => Some(Type::I32),
            "u32" => Some(Type::U32),
            "f32" => Some(Type::F32),
            "f64" => Some(Type::F64),
            "Bool" => Some(Type::Bool),
            "Bytes" | "String" | "Utf8Error" | "ForeignError" => {
                Some(Type::Named(value.to_owned()))
            }
            _ if depth < 4 && parts(value, depth + 1).is_some() => {
                Some(Type::Named(value.to_owned()))
            }
            _ => None,
        }
    }
    fn parts(name: &str, depth: usize) -> Option<(Type, Type)> {
        let inside = name.strip_prefix("Result<")?.strip_suffix('>')?;
        let mut level = 0usize;
        for (index, byte) in inside.bytes().enumerate() {
            match byte {
                b'<' => level = level.checked_add(1)?,
                b'>' => level = level.checked_sub(1)?,
                b',' if level == 0 => {
                    return Some((
                        component(&inside[..index], depth)?,
                        component(&inside[index + 1..], depth)?,
                    ));
                }
                _ => {}
            }
        }
        None
    }
    parts(name, 1)
}

pub(crate) fn view_parts(name: &str) -> Option<(Type, bool)> {
    match name {
        "View<i32>" => Some((Type::I32, false)),
        "View<u32>" => Some((Type::U32, false)),
        "MutView<i32>" => Some((Type::I32, true)),
        "MutView<u32>" => Some((Type::U32, true)),
        _ => None,
    }
}
