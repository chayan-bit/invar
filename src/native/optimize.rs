//! ENG-18 (#329): which CPU an object is generated for, the LLVM optimization
//! pipeline, and the proof-carrying promise rule.
//!
//! Every promise the emitted IR gives LLVM (`noalias`, `inbounds`, `nsw`,
//! `nuw`) must carry `!invar.fact !N`, where node `N` names a [`Fact`] recorded
//! with the IR that proves it. [`audit`] refuses the module before any pass
//! runs otherwise. Promises with no fact form here (other poison-generating
//! flags, fast-math flags, `mustprogress`, `llvm.assume`) are refused outright:
//! fast-math needs a declared #328 contract, and strict never takes
//! `contract` or `reassoc`. Traps stay explicit branches before any use.

use std::ops::RangeInclusive;

use inkwell::passes::PassBuilderOptions;

use super::*;

/// The middle-end pipeline optimized objects run (LLVM's new pass manager).
pub const PIPELINE: &str = "default<O3>";
/// Metadata kind that ties a promise to its fact.
const FACT_KIND: &str = "!invar.fact";
/// Tag of a fact node: `!N = !{!"invar.fact", i64 ID}`, ID indexing the facts.
const FACT_TAG: &str = "!{!\"invar.fact\", i64 ";

/// Which CPU an object's code is generated for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cpu {
    /// LLVM's generic baseline for the host triple: runs on every CPU of the
    /// architecture, so built artifacts stay portable.
    Generic,
    /// The live CPU as LLVM detects it, name and features. An unknown CPU
    /// falls back to the generic baseline.
    Host,
    /// A described CPU from `targets/` (for example `cpu-x86_64-zen4`): its
    /// model selects the LLVM CPU and its `isa` list the features.
    Described(String),
}

/// How [`emit_mir_for`] generates an object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Build {
    pub cpu: Cpu,
    /// Run [`PIPELINE`] before code generation.
    pub optimize: bool,
}

impl Build {
    /// The pre-ENG-18 path: generic CPU, no middle-end passes. Debug and
    /// export objects keep it (#305 variable locations, portable libraries).
    pub const UNOPTIMIZED: Self = Self {
        cpu: Cpu::Generic,
        optimize: false,
    };
    /// Optimized and portable: what `invar build` publishes.
    pub const PORTABLE: Self = Self {
        cpu: Cpu::Generic,
        optimize: true,
    };
    /// Optimized for the live CPU: objects run in-process on this machine.
    pub const HOST: Self = Self {
        cpu: Cpu::Host,
        optimize: true,
    };
}

/// What an object was generated for; part of the native object receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Machine {
    /// LLVM CPU name (`generic`, `znver4`, `apple-m4`, ...).
    pub cpu: String,
    /// LLVM feature string, empty for the baseline.
    pub features: String,
    /// The `targets/` description the CPU matched, if any.
    pub target: Option<String>,
    /// The middle-end pipeline that ran, if any.
    pub pipeline: Option<&'static str>,
}

/// A promise LLVM may exploit: violating it is undefined behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Promise {
    NoAlias,
    InBounds,
    Nsw,
    Nuw,
}

impl Promise {
    fn from_word(word: &str) -> Option<Self> {
        Some(match word {
            "noalias" => Self::NoAlias,
            "inbounds" => Self::InBounds,
            "nsw" => Self::Nsw,
            "nuw" => Self::Nuw,
            _ => return None,
        })
    }
}

/// Exact integer operation an interval fact bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalOp {
    Add,
    Sub,
    Mul,
}

impl IntervalOp {
    fn opcode(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
        }
    }

    fn exact(self, a: i128, b: i128) -> Option<i128> {
        match self {
            Self::Add => a.checked_add(b),
            Self::Sub => a.checked_sub(b),
            Self::Mul => a.checked_mul(b),
        }
    }
}

/// One operand of an interval fact: the IR value it names (`%i`, or a decimal
/// constant, which must lie in its own range) and the closed interval it is in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operand {
    pub value: String,
    pub range: RangeInclusive<i128>,
}

/// Why a promise holds. Facts are checked here, independently of the code
/// that emitted the promise, and each is bound to the site that carries it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fact {
    /// Affine ownership: the `params` marked `noalias` on the definition are
    /// buffers of distinct `owners` (each param one of them), so none overlaps
    /// another pointer the function reaches.
    Disjoint {
        params: Vec<String>,
        owners: Vec<String>,
    },
    /// The `getelementptr inbounds` on `base` stays inside its buffer: `guard`
    /// is the runtime check that ran before the call, and no buffer holds more
    /// than `elements` elements, so offsets cannot wrap the address space.
    InBounds {
        base: String,
        elements: u64,
        guard: String,
    },
    /// `op` on `lhs` and `rhs` at width `bits` has an exact result inside the
    /// signed (`nsw`) or unsigned (`nuw`) range of that width.
    Interval {
        op: IntervalOp,
        bits: u32,
        lhs: Operand,
        rhs: Operand,
    },
}

/// Which pointer parameters a definition line marks `noalias`.
fn noalias_params<'a>(words: &[&'a str]) -> Vec<&'a str> {
    words
        .iter()
        .enumerate()
        .filter(|(_, word)| **word == "noalias")
        .filter_map(|(at, _)| {
            words[at..]
                .iter()
                .find(|word| word.starts_with('%'))
                .copied()
        })
        .collect()
}

impl Fact {
    /// Whether this fact proves `promise` at the IR line split into `words`;
    /// the error says why not.
    fn proves(&self, promise: Promise, words: &[&str]) -> Result<(), String> {
        match (self, promise) {
            (Self::Disjoint { params, owners }, Promise::NoAlias) => {
                let distinct = owners.iter().collect::<std::collections::HashSet<_>>();
                if owners.len() < 2 || distinct.len() != owners.len() {
                    return Err(format!("disjointness needs distinct owners: {owners:?}"));
                }
                if params.is_empty() || params.len() > owners.len() {
                    return Err(format!(
                        "{} noalias params for {} owners",
                        params.len(),
                        owners.len()
                    ));
                }
                let marked = noalias_params(words);
                if words.first() != Some(&"define")
                    || marked.is_empty()
                    || marked
                        .iter()
                        .any(|param| !params.iter().any(|p| p == param))
                {
                    return Err(format!(
                        "disjointness fact covers {params:?}, not {marked:?}"
                    ));
                }
                Ok(())
            }
            (
                Self::InBounds {
                    base,
                    elements,
                    guard,
                },
                Promise::InBounds,
            ) => {
                // Byte offsets of elements up to 16 bytes wide stay inside isize.
                if guard.is_empty() || *elements > isize::MAX as u64 / 16 {
                    return Err(format!("unguarded or oversized bounds fact ({elements})"));
                }
                let gep = words.iter().position(|word| *word == "getelementptr");
                let pointer = gep
                    .filter(|at| words.get(at + 1) == Some(&"inbounds"))
                    .and_then(|at| {
                        words[at..]
                            .iter()
                            .position(|word| *word == "ptr")
                            .map(|p| at + p + 1)
                    })
                    .and_then(|at| words.get(at));
                if pointer != Some(&base.as_str()) {
                    return Err(format!("bounds fact covers {base}, not {pointer:?}"));
                }
                Ok(())
            }
            (Self::Interval { op, bits, lhs, rhs }, Promise::Nsw | Promise::Nuw) => {
                if !(1..=64).contains(bits) || lhs.range.is_empty() || rhs.range.is_empty() {
                    return Err("malformed interval fact".to_owned());
                }
                // `%r = OP [nuw] [nsw] iBITS LHS, RHS`: the fact names this very operation.
                let width = format!("i{bits}");
                let operands = words
                    .iter()
                    .skip(2)
                    .skip_while(|word| matches!(**word, "nsw" | "nuw"))
                    .take(3)
                    .copied()
                    .collect::<Vec<_>>();
                if words.get(1) != Some(&op.opcode())
                    || operands != [width.as_str(), lhs.value.as_str(), rhs.value.as_str()]
                {
                    return Err(format!(
                        "interval fact for {} {width} {}, {} does not name this operation",
                        op.opcode(),
                        lhs.value,
                        rhs.value
                    ));
                }
                for operand in [lhs, rhs] {
                    if let Ok(constant) = operand.value.parse::<i128>()
                        && !operand.range.contains(&constant)
                    {
                        return Err(format!("constant {constant} outside {:?}", operand.range));
                    }
                }
                let (low, high) = if promise == Promise::Nsw {
                    (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
                } else {
                    (0, (1i128 << bits) - 1)
                };
                let inside = |v: i128| (low..=high).contains(&v);
                let (l, r) = (&lhs.range, &rhs.range);
                // Operands must be representable too, or the flag's premise fails.
                let operands = [l.start(), l.end(), r.start(), r.end()]
                    .into_iter()
                    .all(|v| inside(*v));
                // Add, sub and mul are monotone in each operand on a box, so
                // the corners bound the exact result; operands of at most 64
                // bits keep every corner inside i128.
                let corners = [
                    (*l.start(), *r.start()),
                    (*l.start(), *r.end()),
                    (*l.end(), *r.start()),
                    (*l.end(), *r.end()),
                ];
                let results = operands
                    && corners
                        .into_iter()
                        .all(|(a, b)| op.exact(a, b).is_some_and(inside));
                if !results {
                    return Err(format!("interval fact does not prove {promise:?}"));
                }
                Ok(())
            }
            _ => Err(format!("fact {self:?} cannot prove {promise:?}")),
        }
    }
}

/// Words that would give LLVM a promise no fact form covers. Fast-math flags
/// wait for #328 contracts; `contract` and `reassoc` are never strict.
const UNFACTED: &[&str] = &[
    "nusw",
    "exact",
    "disjoint",
    "nneg",
    "samesign",
    "nnan",
    "ninf",
    "nsz",
    "arcp",
    "contract",
    "afn",
    "reassoc",
    "fast",
    "mustprogress",
    "willreturn",
    "nonnull",
    "noundef",
    "dereferenceable",
    "dereferenceable_or_null",
    "@llvm.assume",
];

/// The words of one IR line outside quoted strings, split at whitespace and
/// punctuation. Sigilled names (`%nsw`, `@f`, `!7`) stay single words.
fn words(line: &str) -> Vec<&str> {
    let mut unquoted = Vec::new();
    for (index, part) in line.split('"').enumerate() {
        if index % 2 == 0 {
            // A `;` outside a string starts a comment.
            let (code, comment) = part
                .split_once(';')
                .map_or((part, false), |(code, _)| (code, true));
            unquoted.push(code);
            if comment {
                break;
            }
        }
    }
    unquoted
        .into_iter()
        .flat_map(|part| part.split(|c: char| c.is_whitespace() || "(),{}[]<>=*".contains(c)))
        .filter(|word| !word.is_empty())
        .collect()
}

/// Check every promise in emitted `ir` against `facts`: each must name a fact node
/// whose fact proves it, and nothing outside the fact forms may appear.
pub fn audit(ir: &str, facts: &[Fact]) -> Result<Vec<(Promise, usize)>, String> {
    let mut nodes = HashMap::new();
    for line in ir.lines().filter(|line| line.starts_with('!')) {
        if let Some((name, body)) = line.split_once(" = ")
            && let Some(id) = body
                .strip_prefix(FACT_TAG)
                .and_then(|rest| rest.strip_suffix('}'))
        {
            let id = id
                .parse::<usize>()
                .map_err(|_| format!("malformed fact node {line}"))?;
            nodes.insert(name.trim(), id);
        }
    }
    let mut promised = Vec::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('!') || trimmed.starts_with(';') {
            continue;
        }
        let words = words(trimmed);
        if let Some(word) = words.iter().find(|word| UNFACTED.contains(word)) {
            return Err(format!(
                "native IR contains unproven promise `{word}`: {trimmed}"
            ));
        }
        // Metadata may promise too (`!nonnull`, `!range`, `!noundef`,
        // `!llvm.loop` access groups, ...): only debug locations, debug
        // records' `!DI...` nodes and fact attachments are admitted.
        if let Some(word) = words.iter().find(|word| {
            word.strip_prefix('!').is_some_and(|kind| {
                kind.starts_with(|c: char| c.is_ascii_alphabetic())
                    && kind != "dbg"
                    && **word != FACT_KIND
                    && !kind.starts_with("DI")
            })
        }) {
            return Err(format!(
                "native IR contains unproven promise metadata `{word}`: {trimmed}"
            ));
        }
        let promises = words
            .iter()
            .filter_map(|word| Promise::from_word(word))
            .collect::<Vec<_>>();
        if promises.is_empty() {
            continue;
        }
        let fact = words
            .windows(2)
            .find(|pair| pair[0] == FACT_KIND)
            .and_then(|pair| nodes.get(pair[1]))
            .copied()
            .ok_or_else(|| format!("native IR promise without a fact: {trimmed}"))?;
        let recorded = facts
            .get(fact)
            .ok_or_else(|| format!("native IR names unrecorded fact {fact}: {trimmed}"))?;
        for promise in promises {
            recorded
                .proves(promise, &words)
                .map_err(|why| format!("{why}: {trimmed}"))?;
            promised.push((promise, fact));
        }
    }
    Ok(promised)
}

/// The metadata that ties `facts` to `!invar.fact !N` attachments: node
/// `!{base + id}` for fact `id`. `base` must not collide with other metadata.
pub fn fact_nodes(facts: &[Fact], base: usize) -> String {
    (0..facts.len())
        .map(|id| format!("!{} = {FACT_TAG}{id}}}\n", base + id))
        .collect()
}

/// LLVM CPU and feature names for a described CPU target, or an error naming
/// the description or ISA entry this backend cannot map.
fn described(name: &str, arch: &str) -> Result<(String, String), String> {
    let registry = invar_device::target::Registry::builtin().map_err(|e| e.to_string())?;
    let target = registry
        .get(name)
        .ok_or_else(|| format!("unknown native CPU target {name}"))?;
    let (Some(cpu), Some(target_arch), Some(model)) = (
        target.cpu(),
        target.device.arch.as_deref(),
        target.device.model.as_deref(),
    ) else {
        return Err(format!("{name} does not describe a CPU"));
    };
    if target_arch != arch {
        return Err(format!(
            "{name} is {target_arch}, the host triple is {arch}"
        ));
    }
    let features = cpu
        .isa
        .iter()
        .map(|isa| {
            isa_feature(arch, isa)
                .map(|feature| format!("+{feature}"))
                .ok_or_else(|| format!("{name}: no LLVM feature for isa {isa}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let cpu_name = llvm_cpu(arch, model).unwrap_or("generic");
    Ok((cpu_name.to_owned(), features.join(",")))
}

/// `targets/` model names and the LLVM CPU each selects.
const MODELS: &[(&str, &str, &str)] = &[
    ("x86_64", "zen4", "znver4"),
    ("aarch64", "apple-m4", "apple-m4"),
];

fn llvm_cpu(arch: &str, model: &str) -> Option<&'static str> {
    MODELS
        .iter()
        .find(|(a, m, _)| *a == arch && *m == model)
        .map(|(_, _, cpu)| *cpu)
}

/// The `targets/` description whose model LLVM's `cpu` name selects.
fn model_of(arch: &str, cpu: &str) -> Option<&'static str> {
    MODELS
        .iter()
        .find(|(a, _, c)| *a == arch && *c == cpu)
        .map(|(_, model, _)| *model)
}

/// LLVM feature for a `targets/` ISA name.
fn isa_feature(arch: &str, isa: &str) -> Option<&'static str> {
    Some(match (arch, isa) {
        ("x86_64", "sse4_2") => "sse4.2",
        ("x86_64", "avx2") => "avx2",
        ("x86_64", "fma") => "fma",
        ("x86_64", "avx512f") => "avx512f",
        ("x86_64", "avx512bw") => "avx512bw",
        ("x86_64", "avx512vl") => "avx512vl",
        ("x86_64", "avx512_vnni") => "avx512vnni",
        ("x86_64", "avx512_bf16") => "avx512bf16",
        ("aarch64", "neon") => "neon",
        ("aarch64", "fp16") => "fullfp16",
        ("aarch64", "dotprod") => "dotprod",
        ("aarch64", "i8mm") => "i8mm",
        ("aarch64", "bf16") => "bf16",
        ("aarch64", "sme") => "sme",
        ("aarch64", "sme2") => "sme2",
        _ => return None,
    })
}

/// The architecture field of an LLVM triple, in `targets/` spelling.
fn triple_arch(triple: &str) -> &str {
    match triple.split('-').next().unwrap_or("") {
        "arm64" => "aarch64",
        arch => arch,
    }
}

/// Resolve `cpu` for `triple` to the LLVM CPU name, features and matched
/// description.
pub fn resolve(cpu: &Cpu, triple: &str) -> Result<Machine, String> {
    let arch = triple_arch(triple);
    let (name, features, target) = match cpu {
        Cpu::Generic => ("generic".to_owned(), String::new(), None),
        Cpu::Described(target) => {
            let (name, features) = described(target, arch)?;
            (name, features, Some(target.clone()))
        }
        Cpu::Host => {
            let name = TargetMachine::get_host_cpu_name().to_string();
            if name.is_empty() || name == "generic" {
                ("generic".to_owned(), String::new(), None)
            } else {
                let features = TargetMachine::get_host_cpu_features().to_string();
                // Name the description only if the live CPU has every feature
                // it lists (SKUs may disable some). LLVM reports no feature
                // string on some hosts (Apple silicon); there the CPU name
                // alone selects the features, so the model match decides.
                let target = model_of(arch, &name)
                    .map(|model| format!("cpu-{arch}-{model}"))
                    .filter(|target| {
                        described(target, arch).is_ok_and(|(_, wanted)| {
                            features.is_empty()
                                || wanted
                                    .split(',')
                                    .all(|feature| features.split(',').any(|live| live == feature))
                        })
                    });
                (name, features, target)
            }
        }
    };
    Ok(Machine {
        cpu: name,
        features,
        target,
        pipeline: None,
    })
}

/// Audit, optimize and generate an object for `data` as `build` asks.
pub(super) fn object(data: NativeIr, facts: Vec<Fact>, build: &Build) -> Result<Object, String> {
    let (ir, names, blobs, foreign) = data;
    // The emitted text is exactly what this backend promises; LLVM's parser
    // adds only its own intrinsic attributes.
    audit(&ir, &facts)?;
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|error| error.to_string())?;
    let mut receipt = resolve(&build.cpu, &triple.as_str().to_string_lossy())?;
    let level = if build.optimize {
        OptimizationLevel::Aggressive
    } else {
        OptimizationLevel::Default
    };
    let machine = target
        .create_target_machine(
            &triple,
            &receipt.cpu,
            &receipt.features,
            level,
            // The host links as a position-independent executable, and the
            // invar-trig-f32-v1 helpers address local constant tables, so the
            // object is PIC on every target (x86_64 rejected R_X86_64_32S).
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("LLVM could not create the native target machine")?;
    let context = Context::create();
    let buffer = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "invar-native-v1");
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| error.to_string())?;
    let layout = machine.get_target_data().get_data_layout();
    module.set_triple(&triple);
    module.set_data_layout(&layout);
    module.verify().map_err(|error| error.to_string())?;
    let rendered = module.print_to_string().to_string();
    let optimized = if build.optimize {
        let options = PassBuilderOptions::create();
        options.set_loop_vectorization(true);
        options.set_loop_slp_vectorization(true);
        options.set_loop_interleaving(true);
        options.set_loop_unrolling(true);
        module
            .run_passes(PIPELINE, &machine, options)
            .map_err(|error| error.to_string())?;
        module.verify().map_err(|error| error.to_string())?;
        receipt.pipeline = Some(PIPELINE);
        Some(module.print_to_string().to_string())
    } else {
        None
    };
    let bytes = machine
        .write_to_memory_buffer(&module, FileType::Object)
        .map_err(|error| error.to_string())?
        .as_slice()
        .to_vec();
    Ok(Object {
        triple: triple.as_str().to_string_lossy().into_owned(),
        data_layout: layout.as_str().to_string_lossy().into_owned(),
        ir: rendered,
        optimized_ir: optimized,
        bytes,
        names,
        blobs,
        foreign,
        debug: None,
        machine: receipt,
        facts,
    })
}

/// Test and evidence hook: compile hand-written IR with its facts, returning
/// the object and the target assembly (vectorization evidence, #329).
#[doc(hidden)]
pub fn compile_ir(ir: &str, facts: Vec<Fact>, build: &Build) -> Result<(Object, String), String> {
    let object = object(
        (ir.to_owned(), Vec::new(), Vec::new(), Vec::new()),
        facts,
        build,
    )?;
    let assembly = assembly(&object)?;
    Ok((object, assembly))
}

/// Target assembly of an object's final IR, for the same machine.
fn assembly(object: &Object) -> Result<String, String> {
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).map_err(|error| error.to_string())?;
    let machine = target
        .create_target_machine(
            &triple,
            &object.machine.cpu,
            &object.machine.features,
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or("LLVM could not create the native target machine")?;
    let context = Context::create();
    let text = object.optimized_ir.as_deref().unwrap_or(&object.ir);
    let buffer = MemoryBuffer::create_from_memory_range_copy(text.as_bytes(), "invar-native-v1");
    let module = context
        .create_module_from_ir(buffer)
        .map_err(|error| error.to_string())?;
    let buffer = machine
        .write_to_memory_buffer(&module, FileType::Assembly)
        .map_err(|error| error.to_string())?;
    Ok(String::from_utf8_lossy(buffer.as_slice()).into_owned())
}
