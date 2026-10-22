//! Source fixtures reach the real live ceiling with individually legal arrays.
use invar::array_limits::{MAX_ARRAY_LEN, MAX_LIVE_ELEMENTS};

pub fn padding(elements: usize) -> String {
    assert!(elements <= MAX_LIVE_ELEMENTS);
    let mut source = String::new();
    let mut remaining = elements;
    for index in 0..elements.div_ceil(MAX_ARRAY_LEN) {
        let length = remaining.min(MAX_ARRAY_LEN);
        source.push_str(&format!(
            "let limit_pad_{index} = filled_u32({length}u32,0u32);"
        ));
        remaining -= length;
    }
    source
}
