//! Seeded generator of well-typed GNET-02 source compositions. Fragments are
//! independent statements over `var acc: i32` with uniquely indexed names.

pub const SEED: u64 = 0x6e65_7402_5eed_0001;
pub const PROGRAMS: usize = 240;

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*: deterministic, no dependency.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, low: i64, high: i64) -> i64 {
        low + self.below((high - low + 1) as u64) as i64
    }
}

/// One fragment: top-level declarations and a statement over `var acc: i32`.
fn fragment(rng: &mut Rng, i: usize) -> (String, String, &'static str) {
    let small = |rng: &mut Rng| rng.range(-40, 40);
    match rng.below(16) {
        0 => {
            let big = rng.below(6) == 0;
            let (a, b) = if big {
                (rng.range(30_000, 60_000), rng.range(30_000, 60_000))
            } else {
                (small(rng), small(rng))
            };
            (String::new(), format!("acc = acc + {a} * {b};"), "arith")
        }
        1 => {
            let (d, e) = (rng.range(0, 3), rng.range(0, 3));
            (
                String::new(),
                format!("acc = acc / ({d} - {e}) + acc % ({d} + 1);"),
                "div_rem",
            )
        }
        2 => (
            String::new(),
            format!(
                "if acc % 2 == 0 {{ acc = acc + {}; }} else {{ acc = acc - {}; }}",
                small(rng),
                small(rng)
            ),
            "branch",
        ),
        3 => (
            String::new(),
            format!(
                "for i{i} in 0u32..{}u32 {{ acc = acc + {}; }}",
                rng.below(6),
                small(rng)
            ),
            "for",
        ),
        4 => (
            String::new(),
            format!(
                "var w{i} = {}; while w{i} > 0 {{ w{i} = w{i} - 1; acc = acc + {}; }}",
                rng.below(5),
                small(rng)
            ),
            "while",
        ),
        5 => {
            let depth = if rng.below(4) == 0 {
                rng.range(60, 66)
            } else {
                rng.range(0, 8)
            };
            (
                format!(
                    "fn r{i}(n: u32) -> i32 {{ if n == 0u32 {{ return {}; }} return r{i}(n - 1u32) + {}; }}",
                    small(rng),
                    small(rng)
                ),
                format!("acc = acc + r{i}({depth}u32);"),
                "recursion",
            )
        }
        6 => {
            let n = rng.range(1, 4);
            (
                String::new(),
                format!(
                    "var xs{i}: Array<i32> = filled_i32({n}u32, {}); xs{i}[{}u32] = acc; acc = acc + xs{i}[{}u32];",
                    small(rng),
                    rng.range(0, n),
                    rng.range(0, n)
                ),
                "array",
            )
        }
        7 => (
            format!(
                "struct P{i} {{ items: Array<i32>, code: i32 }} fn s{i}(p: read P{i}) -> i32 {{ return p.code + {}; }}",
                small(rng)
            ),
            format!(
                "let p{i}: P{i} = P{i} {{ items: [{}, {}], code: {} }}; acc = acc + s{i}(p{i});",
                small(rng),
                small(rng),
                small(rng)
            ),
            "record",
        ),
        8 => {
            let value = if rng.below(2) == 0 {
                format!("E{i}::Value(acc)")
            } else {
                format!("E{i}::Empty()")
            };
            (
                format!(
                    "enum E{i} {{ Empty, Value(i32) }} fn m{i}(e: read E{i}) -> i32 {{ return match e {{ E{i}::Empty => {}, E{i}::Value(x{i}) => x{i} + {} }}; }}",
                    small(rng),
                    small(rng)
                ),
                format!("let e{i}: E{i} = {value}; acc = acc + m{i}(e{i});"),
                "variant",
            )
        }
        9 => {
            let (start, end) = (rng.range(0, 7), rng.range(0, 7));
            let stride = if rng.below(8) == 0 {
                0
            } else {
                rng.range(1, 3)
            };
            (
                String::new(),
                format!(
                    "var v{i}: Array<i32> = [1, 2, 3, 4, 5, 6]; if true {{ let mv{i} = view_mut(v{i}, {start}u32, {end}u32, {stride}u32); if len(mv{i}) > 0u32 {{ mv{i}[0u32] = acc; }} }} acc = acc + v{i}[{}u32];",
                    rng.range(0, 6)
                ),
                "view",
            )
        }
        10 => (
            String::new(),
            format!(
                "let b{i}: Bytes = b\"abcd\"; let c{i}: Bytes = bytes_slice(b{i}, {}u32, {}u32); if len(c{i}) == {}u32 {{ acc = acc + 1; }}",
                rng.range(0, 5),
                rng.range(0, 5),
                rng.range(0, 3)
            ),
            "bytes",
        ),
        11 => (
            String::new(),
            format!(
                "if acc > {} {{ let t{i}: Array<u32> = [1u32, 2u32, 3u32]; acc = acc + 1; }} else {{ let f{i}: Array<i32> = [4]; acc = acc - 1; }}",
                small(rng)
            ),
            "scoped_cleanup",
        ),
        12 => (
            String::new(),
            format!("if acc > {} {{ return acc; }}", rng.range(-50, 500)),
            "early_return",
        ),
        13 => (
            String::new(),
            "if false { acc = acc / 0; } if acc == acc or (1 / 0) == 1 { acc = acc + 1; }"
                .to_owned(),
            "untaken_trap",
        ),
        14 => {
            let raw = if rng.below(2) == 0 {
                "caf\\xC3\\xA9"
            } else {
                "caf\\xC3"
            };
            (
                String::new(),
                format!(
                    "let raw{i}: Bytes = b\"{raw}\"; let d{i} = utf8_decode(raw{i}); acc = acc + match move d{i} {{ Result::Ok(text{i}) => {}, Result::Err(err{i}) => {} }};",
                    small(rng),
                    small(rng)
                ),
                "utf8_result",
            )
        }
        _ => (
            String::new(),
            format!(
                "let z{i}: u32 = wrapping_mul({}u32, {}u32); if z{i} > {}u32 {{ acc = acc + 1; }}",
                rng.next() as u32,
                rng.next() as u32,
                rng.next() as u32
            ),
            "wrapping",
        ),
    }
}

pub fn program(rng: &mut Rng) -> (String, Vec<&'static str>) {
    let mut decls = String::new();
    let mut body = String::new();
    let mut kinds = Vec::new();
    for i in 0..rng.range(2, 8) as usize {
        let (decl, stmt, kind) = fragment(rng, i);
        decls.push_str(&decl);
        decls.push(' ');
        body.push_str(&stmt);
        body.push(' ');
        kinds.push(kind);
    }
    let seed = rng.range(-20, 20);
    (
        format!("{decls}fn main() -> i32 {{ var acc = {seed}; {body}return acc; }}"),
        kinds,
    )
}
