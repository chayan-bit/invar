// The internal i64 carrier holds binary32 bits (zero-extended) or binary64
// bits; there is no new public ABI tag. Math results are canonical NaNs.

/// LLVM spelling of one IEEE format: (IR type, intrinsic suffix, CORE-MATH
/// suffix, canonical quiet NaN as IR hex).
#[derive(Clone, Copy)]
struct FloatFormat {
    ir: &'static str,
    intrinsic: &'static str,
    core_math: &'static str,
    nan: &'static str,
}

const BINARY32: FloatFormat = FloatFormat {
    ir: "float",
    intrinsic: "f32",
    core_math: "f",
    nan: "0x7FF8000000000000",
};
const BINARY64: FloatFormat = FloatFormat {
    ir: "double",
    intrinsic: "f64",
    core_math: "",
    nan: "0x7FF8000000000000",
};

fn float_format(ty: Option<&Type>) -> Option<FloatFormat> {
    match ty {
        Some(Type::F32) => Some(BINARY32),
        Some(Type::F64) => Some(BINARY64),
        _ => None,
    }
}

/// Declarations every non-exporting module may call: explicit fma, the exact
/// `invar-math-cr-v1` operations and the CORE-MATH entry points.
fn float_declarations() -> String {
    let mut text = String::from("declare i32 @invar_math_hold_nearest()\ndeclare void @invar_math_restore_rounding(i32)\n");
    for format in [BINARY32, BINARY64] {
        let (t, s) = (format.ir, format.intrinsic);
        text.push_str(&format!("declare {t} @llvm.fma.{s}({t}, {t}, {t})\n"));
        for unary in ["sqrt", "floor", "ceil", "trunc", "round", "fabs"] {
            text.push_str(&format!("declare {t} @llvm.{unary}.{s}({t})\n"));
        }
        text.push_str(&format!("declare {t} @llvm.copysign.{s}({t}, {t})\n"));
        for function in invar_math::Function::ALL {
            let params = if function.binary() { format!("{t}, {t}") } else { t.to_owned() };
            text.push_str(&format!(
                "declare {t} @cr_{}{}({params}) nounwind\n",
                function.name(),
                format.core_math
            ));
        }
    }
    text
}

impl Ir<'_> {
    fn decode_float(&mut self, format: FloatFormat, word: &str) -> String {
        let decoded = self.tmp();
        if format.intrinsic == "f64" {
            self.line(format!("{decoded} = bitcast i64 {word} to double"));
        } else {
            let bits = self.tmp();
            self.line(format!("{bits} = trunc i64 {word} to i32"));
            self.line(format!("{decoded} = bitcast i32 {bits} to float"));
        }
        decoded
    }

    fn encode_float(&mut self, format: FloatFormat, value: &str) -> String {
        let word = self.tmp();
        if format.intrinsic == "f64" {
            self.line(format!("{word} = bitcast double {value} to i64"));
        } else {
            let bits = self.tmp();
            self.line(format!("{bits} = bitcast float {value} to i32"));
            self.line(format!("{word} = zext i32 {bits} to i64"));
        }
        word
    }

    fn binary_float(&mut self, format: FloatFormat, op: BinaryOp, a: &str, b: &str) -> Result<String, String> {
        use BinaryOp as O;
        let t = format.ir;
        let a = self.decode_float(format, a);
        let b = self.decode_float(format, b);
        let value = self.tmp();
        let predicate = match op {
            O::Eq => Some("oeq"),
            O::Ne => Some("une"),
            O::Lt => Some("olt"),
            O::Le => Some("ole"),
            O::Gt => Some("ogt"),
            O::Ge => Some("oge"),
            _ => None,
        };
        if let Some(predicate) = predicate {
            let word = self.tmp();
            self.line(format!("{value} = fcmp {predicate} {t} {a}, {b}"));
            self.line(format!("{word} = zext i1 {value} to i64"));
            return Ok(word);
        }
        let opcode = match op {
            O::Add => "fadd",
            O::Sub => "fsub",
            O::Mul => "fmul",
            O::Div => "fdiv",
            _ => return Err(error("verified float operator is unsupported")),
        };
        // Strict ordered policy: no fast/reassoc/contract flags. Explicit fma is separate.
        self.line(format!("{value} = {opcode} {t} {a}, {b}"));
        Ok(self.encode_float(format, &value))
    }

    fn negate_float(&mut self, format: FloatFormat, word: &str) -> String {
        let decoded = self.decode_float(format, word);
        let negated = self.tmp();
        self.line(format!("{negated} = fneg {} {decoded}", format.ir));
        self.encode_float(format, &negated)
    }

    /// Replace any NaN by the canonical quiet NaN of `format`.
    fn canonical(&mut self, format: FloatFormat, value: &str) -> String {
        let (t, nan) = (format.ir, format.nan);
        let unordered = self.tmp();
        let chosen = self.tmp();
        self.line(format!("{unordered} = fcmp uno {t} {value}, {value}"));
        self.line(format!("{chosen} = select i1 {unordered}, {t} {nan}, {t} {value}"));
        chosen
    }

    /// IEEE 754-2019 minimumNumber (or maximumNumber) spelled out in plain
    /// IR: a NaN operand loses, and equal operands merge their sign bits so
    /// -0 is below +0. `llvm.minimumnum` is not used because its x86 lowering
    /// does not order signed zeros for f64 (LLVM 21: min(-0.0, 0.0) gave +0).
    fn minimum_maximum_number(&mut self, format: FloatFormat, maximum: bool, a: &str, b: &str) -> String {
        let t = format.ir;
        let int = if format.intrinsic == "f64" { "i64" } else { "i32" };
        let (order, merge) = if maximum { ("ogt", "and") } else { ("olt", "or") };
        let [a_nan, b_nan, a_wins, b_wins] = [(); 4].map(|()| self.tmp());
        let [a_bits, b_bits, merged_bits, merged, tie, ordered, unless_b_nan, chosen] = [(); 8].map(|()| self.tmp());
        self.line(format!("{a_nan} = fcmp uno {t} {a}, {a}"));
        self.line(format!("{b_nan} = fcmp uno {t} {b}, {b}"));
        self.line(format!("{a_wins} = fcmp {order} {t} {a}, {b}"));
        self.line(format!("{b_wins} = fcmp {order} {t} {b}, {a}"));
        self.line(format!("{a_bits} = bitcast {t} {a} to {int}"));
        self.line(format!("{b_bits} = bitcast {t} {b} to {int}"));
        self.line(format!("{merged_bits} = {merge} {int} {a_bits}, {b_bits}"));
        self.line(format!("{merged} = bitcast {int} {merged_bits} to {t}"));
        self.line(format!("{tie} = select i1 {b_wins}, {t} {b}, {t} {merged}"));
        self.line(format!("{ordered} = select i1 {a_wins}, {t} {a}, {t} {tie}"));
        self.line(format!("{unless_b_nan} = select i1 {b_nan}, {t} {a}, {t} {ordered}"));
        self.line(format!("{chosen} = select i1 {a_nan}, {t} {b}, {t} {unless_b_nan}"));
        chosen
    }

    /// One `invar-math-cr-v1` function on decoded operands, as an i64 word.
    fn math_call(&mut self, function: crate::math::MathFn, format: FloatFormat, args: &[String]) -> Result<String, String> {
        use crate::math::MathFn as M;
        let (t, s) = (format.ir, format.intrinsic);
        let mut decoded = args
            .iter()
            .map(|word| self.decode_float(format, word))
            .collect::<Vec<_>>();
        if decoded.len() != function.arity() {
            return Err(error("verified math arity mismatch"));
        }
        if self.exporting {
            return Err(error("native exports do not link the invar-math-cr-v1 library"));
        }
        if function == M::Copysign {
            // A NaN sign source counts as positive.
            let unordered = self.tmp();
            let sign = self.tmp();
            self.line(format!("{unordered} = fcmp uno {t} {}, {}", decoded[1], decoded[1]));
            self.line(format!("{sign} = select i1 {unordered}, {t} 0.0, {t} {}", decoded[1]));
            decoded[1] = sign;
        }
        if matches!(function, M::Min | M::Max) {
            let chosen = self.minimum_maximum_number(format, function == M::Max, &decoded[0], &decoded[1]);
            let result = self.canonical(format, &chosen);
            return Ok(self.encode_float(format, &result));
        }
        let callee = match (function, crate::math::core_math(function)) {
            (_, Some(core)) => format!("cr_{}{}", core.name(), format.core_math),
            (M::Abs, None) => format!("llvm.fabs.{s}"),
            (other, None) => format!("llvm.{}.{s}", other.name()),
        };
        let operands = decoded
            .iter()
            .map(|value| format!("{t} {value}"))
            .collect::<Vec<_>>()
            .join(", ");
        let raw = self.tmp();
        if crate::math::core_math(function).is_some() {
            // CORE-MATH honours the ambient rounding mode; hold round-to-nearest
            // around the call and restore the caller's mode (crates/invar-math/src/nearest.c).
            let mode = self.tmp();
            self.line(format!("{mode} = call i32 @invar_math_hold_nearest()"));
            self.line(format!("{raw} = call {t} @{callee}({operands})"));
            self.line(format!("call void @invar_math_restore_rounding(i32 {mode})"));
        } else {
            self.line(format!("{raw} = call {t} @{callee}({operands})"));
        }
        let result = self.canonical(format, &raw);
        Ok(self.encode_float(format, &result))
    }

    /// `name` lowered when it is a float intrinsic; `None` otherwise.
    fn float_intrinsic(&mut self, name: &str, args: &[String], arg_types: &[Option<Type>], at: Span) -> Result<Option<String>, String> {
        let first = float_format(arg_types.first().and_then(Option::as_ref));
        if let Some(function) = crate::math::MathFn::from_name(name) {
            let format = first.ok_or_else(|| error("verified math operand type missing"))?;
            return self.math_call(function, format, args).map(Some);
        }
        if let ("fma", Some(format)) = (name, first) {
            let t = format.ir;
            let [a, b, c] = args else { return Err(error("verified fma arity mismatch")) };
            let (a, b, c) = (self.decode_float(format, a), self.decode_float(format, b), self.decode_float(format, c));
            let fused = self.tmp();
            self.line(format!("{fused} = call {t} @llvm.fma.{}({t} {a}, {t} {b}, {t} {c})", format.intrinsic));
            return Ok(Some(self.encode_float(format, &fused)));
        }
        if matches!(name, "store-f16" | "store-bf16" | "store-i8" | "store-u8") {
            return self.storage_store(name, args, at).map(Some);
        }
        let Some((input, output)) = crate::check::conversion(name) else { return Ok(None) };
        let [word] = args else { return Err(error("verified conversion arity mismatch")) };
        let target = float_format(Some(&output));
        let converted = match (float_format(Some(&input)), target) {
            (None, Some(format)) => {
                let narrow = self.tmp();
                let value = self.tmp();
                let opcode = if input == Type::I32 { "sitofp" } else { "uitofp" };
                self.line(format!("{narrow} = trunc i64 {word} to i32"));
                self.line(format!("{value} = {opcode} i32 {narrow} to {}", format.ir));
                self.encode_float(format, &value)
            }
            (Some(from), Some(to)) => {
                let decoded = self.decode_float(from, word);
                let value = self.tmp();
                let opcode = if to.intrinsic == "f64" { "fpext" } else { "fptrunc" };
                self.line(format!("{value} = {opcode} {} {decoded} to {}", from.ir, to.ir));
                let value = self.canonical(to, &value);
                self.encode_float(to, &value)
            }
            (Some(from), None) => self.float_to_integer(from, word, output == Type::I32, at)?,
            (None, None) => return Err(error("verified conversion is not a float conversion")),
        };
        Ok(Some(converted))
    }

    /// A store into a storage array (`check::storage`, ENG-03): f16 and bf16
    /// round once to nearest even (a NaN stays the NaN class), i8 and u8 trap
    /// INTEGER_OVERFLOW out of range.
    fn storage_store(&mut self, name: &str, args: &[String], at: Span) -> Result<String, String> {
        let [word] = args else { return Err(error("verified store arity mismatch")) };
        let format = float_format(Some(&Type::F32)).ok_or_else(|| error("binary32 format missing"))?;
        let rounded = match name {
            "store-f16" => {
                let x = self.decode_float(format, word);
                let (half, back) = (self.tmp(), self.tmp());
                self.line(format!("{half} = fptrunc float {x} to half"));
                self.line(format!("{back} = fpext half {half} to float"));
                back
            }
            "store-bf16" => {
                // Add half a bfloat16 ulp, less one unless the kept bit is odd, then cut.
                let x = self.decode_float(format, word);
                let [bits, high, odd, bias, sum, cut, value, nan, chosen] = [(); 9].map(|_| self.tmp());
                self.line(format!("{bits} = trunc i64 {word} to i32"));
                self.line(format!("{high} = lshr i32 {bits}, 16"));
                self.line(format!("{odd} = and i32 {high}, 1"));
                self.line(format!("{bias} = add i32 {odd}, 32767"));
                self.line(format!("{sum} = add i32 {bits}, {bias}"));
                self.line(format!("{cut} = and i32 {sum}, -65536"));
                self.line(format!("{value} = bitcast i32 {cut} to float"));
                self.line(format!("{nan} = fcmp uno float {x}, {x}"));
                self.line(format!("{chosen} = select i1 {nan}, float {x}, float {value}"));
                chosen
            }
            _ => {
                let [value, inside, bad] = [(); 3].map(|_| self.tmp());
                self.line(format!("{value} = trunc i64 {word} to i32"));
                if name == "store-i8" {
                    let [above, below] = [(); 2].map(|_| self.tmp());
                    self.line(format!("{above} = icmp sge i32 {value}, -128"));
                    self.line(format!("{below} = icmp sle i32 {value}, 127"));
                    self.line(format!("{inside} = and i1 {above}, {below}"));
                } else {
                    self.line(format!("{inside} = icmp ule i32 {value}, 255"));
                }
                self.line(format!("{bad} = xor i1 {inside}, true"));
                self.trap_if(&bad, 1, at)?;
                return Ok(word.clone());
            }
        };
        let value = self.canonical(format, &rounded);
        Ok(self.encode_float(format, &value))
    }

    /// Truncate toward zero; NaN or out of range traps INTEGER_OVERFLOW.
    fn float_to_integer(&mut self, from: FloatFormat, word: &str, signed: bool, at: Span) -> Result<String, String> {
        let mut value = self.decode_float(from, word);
        if from.intrinsic == "f32" {
            let wide = self.tmp();
            self.line(format!("{wide} = fpext float {value} to double"));
            value = wide;
        }
        let truncated = self.tmp();
        self.line(format!("{truncated} = call double @llvm.trunc.f64(double {value})"));
        let (low, high) = if signed {
            ("-2147483648.0", "2147483647.0")
        } else {
            ("0.0", "4294967295.0")
        };
        let above = self.tmp();
        let below = self.tmp();
        let inside = self.tmp();
        let bad = self.tmp();
        self.line(format!("{above} = fcmp oge double {truncated}, {low}"));
        self.line(format!("{below} = fcmp ole double {truncated}, {high}"));
        self.line(format!("{inside} = and i1 {above}, {below}"));
        self.line(format!("{bad} = xor i1 {inside}, true"));
        self.trap_if(&bad, 1, at)?;
        let integer = self.tmp();
        let word = self.tmp();
        let (opcode, extend) = if signed { ("fptosi", "sext") } else { ("fptoui", "zext") };
        self.line(format!("{integer} = {opcode} double {truncated} to i32"));
        self.line(format!("{word} = {extend} i32 {integer} to i64"));
        Ok(word)
    }
}
