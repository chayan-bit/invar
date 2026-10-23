//! #54: the source/MIR recognizers as test oracles, and the program corpus
//! selection parity is measured on. A recognizer selects; the production
//! dispatcher must make the same choice from the checked normal form, or name
//! why it refused.
#![allow(dead_code)]

use invar::{admission, check, front, mir, plan::PlanKind};
use std::path::{Path, PathBuf};

/// Every family, by module. Each one's production selection is a normal-form
/// lowering; its recognizer survives only here, as the oracle.
pub const FAMILIES: &[&str] = &["core", "tensor", "grid", "segmented", "indexed"];

/// One recognizer choice: function, loop span, family module.
pub type Choice = (String, front::Span, &'static str);

/// The recognizer chain in `families!` precedence order: the first family
/// whose recognizer accepts an A0 loop owns it.
pub fn oracle(program: &Program) -> Vec<Choice> {
    let report = admission::analyze(&program.source, &program.mir).expect("A0 admission");
    let mut out = Vec::new();
    for region in report.regions() {
        let (function, span) = (region.function(), region.span());
        let family = if region.plan_kind() != PlanKind::Serial {
            "core"
        } else if tensor(program, function, span) {
            "tensor"
        } else if grid(program, function, span) {
            "grid"
        } else if segmented(&program.source, &program.mir, function, span) {
            "segmented"
        } else if indexed(&program.source, &program.mir, function, span) {
            "indexed"
        } else {
            continue;
        };
        out.push((function.to_owned(), span, family));
    }
    out
}

/// The tensor recognizer: a scalar contraction annotated on the program at
/// this loop whose source extraction and compiled MIR are the program's, else
/// a direct-output contraction whose loop is this one and whose MIR verifies.
/// A mismatch fails the program in production, so it is no match.
fn tensor(program: &Program, function: &str, span: front::Span) -> bool {
    let annotated = &program.mir.tensor_contracts;
    if let Some(contract) = annotated
        .iter()
        .find(|c| c.function() == function && c.loop_span() == span)
    {
        let digest = program.mir.contract_digest();
        return front::parse(&program.source)
            .ok()
            .and_then(|p| check::check(p, false).ok())
            .and_then(|checked| invar::tensor::Contraction::from_checked(&checked, function).ok())
            .is_some_and(|actual| {
                actual.contract() == contract
                    && actual
                        .compiled_mir()
                        .is_ok_and(|m| m.contract_digest() == digest)
            });
    }
    invar::contraction_source::Contract::from_source(&program.source, function)
        .is_ok_and(|c| c.loop_span() == span && c.verify_mir(&program.mir).is_ok())
}

/// The grid recognizer: a transpose, else a stencil, source contract of a
/// three-parameter `function` whose loop is this one, bound to the program's
/// MIR by its function fingerprint. A contract whose fingerprint differs fails
/// the program in production (`GRID_MIR_MISMATCH`), so it is no match here.
fn grid(program: &Program, function: &str, span: front::Span) -> bool {
    let Some(checked) = front::parse(&program.source)
        .ok()
        .and_then(|p| check::check(p, true).ok())
    else {
        return false;
    };
    if !checked
        .program
        .functions
        .iter()
        .any(|f| f.name == function && f.params.len() == 3)
    {
        return false;
    }
    let contract = invar::transpose_source::Contract::from_checked(&checked, function)
        .ok()
        .filter(|c| c.loop_span() == span)
        .map(|c| c.function_contract())
        .or_else(|| {
            invar::stencil_source::Contract::from_checked(&checked, function)
                .ok()
                .filter(|c| c.loop_span() == span)
                .map(|c| c.function_contract())
        });
    contract.is_some_and(|contract| {
        program
            .mir
            .functions
            .iter()
            .any(|f| f.name == function && fingerprint(f) == contract)
    })
}

/// The digest a grid contract binds its function by: SHA-256 of the debug
/// form of the function's fields, locals sorted by name (as production's
/// `tensor::function_fingerprint`).
fn fingerprint(function: &mir::Function) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut locals = function.locals.iter().collect::<Vec<_>>();
    locals.sort_by_key(|(name, _)| *name);
    let fields = format!(
        "{:?}",
        (
            &function.name,
            &function.foreign_oracle,
            &function.params,
            &function.result,
            &function.return_from,
            function.span,
            function.entry,
            &function.blocks,
            locals,
            &function.declaration_order
        )
    );
    Sha256::digest(fields.as_bytes()).into()
}

/// The segmented recognizer: the function's segmented map contract decides
/// alone when it has one; else its segmented reduction; else one of the two
/// compaction passes. A contract whose MIR differs from the program's, or
/// any extraction error, is not a match (production fails the program).
fn segmented(source: &[u8], program: &mir::Program, function: &str, span: front::Span) -> bool {
    use invar::{compaction_source as cs, segmented_device as sd, segmented_source as ss};
    let digest = program.contract_digest();
    match sd::MapContract::from_source(source, function) {
        Ok(c) => return c.mir_contract() == digest && c.loop_span() == span,
        Err(sd::Error::Unrecognized) => {}
        Err(_) => return false,
    }
    match ss::Contract::from_source(source, function) {
        Ok(c) => return c.verify_mir(program).is_ok() && c.loop_span() == span,
        Err(ss::Error::Unrecognized) => {}
        Err(_) => return false,
    }
    cs::Contract::from_source(source, function)
        .is_ok_and(|c| c.verify_mir(program).is_ok() && c.loop_spans().contains(&span))
}

/// The indexed recognizer: the first of the gather/injective-scatter, affine
/// scatter, histogram and frontier source contracts `function` has, when its
/// loop is this one, its verified MIR is the program's, and an affine scatter
/// starts at zero.
fn indexed(source: &[u8], program: &mir::Program, function: &str, span: front::Span) -> bool {
    use invar::{
        frontier_source as fr, histogram_source as hs, indexed_source as ix, scatter_proof as sp,
    };
    let Some(checked) = front::parse(source)
        .ok()
        .and_then(|p| check::check(p, false).ok())
    else {
        return false;
    };
    let digest = |m: Option<mir::Program>| m.map(|m| m.contract_digest());
    let (own, verified, zero) = match ix::Contract::from_checked(source, &checked, function) {
        Ok(c) => (c.loop_span(), digest(c.verified_mir().ok()), true),
        Err(ix::Error::Unrecognized) => {
            match sp::Contract::from_checked(source, &checked, function) {
                Ok(c) => {
                    let cert = c.certificate();
                    (
                        cert.loop_span(),
                        digest(c.verified_mir().ok()),
                        cert.domain().0 == 0,
                    )
                }
                Err(sp::Error::Unrecognized) => {
                    match hs::Contract::from_checked(source, &checked, function) {
                        Ok(c) => (c.loop_span(), digest(c.verified_mir().ok()), true),
                        Err(hs::Error::Unrecognized) => {
                            match fr::Contract::from_checked(source, &checked, function) {
                                Ok(c) => (c.loop_span(), digest(c.verified_mir().ok()), true),
                                Err(_) => return false,
                            }
                        }
                        Err(_) => return false,
                    }
                }
                Err(_) => return false,
            }
        }
        Err(_) => return false,
    };
    own == span && zero && verified == Some(program.contract_digest())
}

pub struct Program {
    pub id: String,
    pub source: Vec<u8>,
    pub checked: check::CheckedProgram,
    pub mir: mir::Program,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries = std::fs::read_dir(dir)
        .expect("directory")
        .map(|e| e.expect("entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        let name = path.file_name().unwrap().to_string_lossy();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                walk(&path, out);
            }
        } else if name.ends_with(".inv") {
            out.push(path);
        }
    }
}

/// The whole corpus: the examples, the benchmark corpora and the test fixtures.
pub const CORPUS: &[&str] = &["examples", "bench", "tools", "tests", "demo"];

/// Every tracked `.inv` program under `dirs` with a `main` that checks, lowers
/// and has an A0 report, one per distinct source, mapped through `f` on up to
/// `THREADS` threads.
pub fn each<T: Send>(dirs: &[&str], f: impl Fn(&Program) -> T + Sync) -> Vec<T> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    for dir in dirs {
        if root.join(dir).is_dir() {
            walk(&root.join(dir), &mut paths);
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    let sources: Vec<_> = paths
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read(&path).unwrap();
            let id = path.strip_prefix(&root).unwrap().display().to_string();
            (String::from_utf8_lossy(&source).contains("fn main") && seen.insert(source.clone()))
                .then_some((id, source))
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<(usize, T)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                // The largest corpus programs overflow the default 2 MiB stack
                // in an unoptimized (dev profile) build.
                let worker = std::thread::Builder::new().stack_size(STACK);
                worker
                    .spawn_scoped(scope, || {
                        let mut mine = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some((id, source)) = sources.get(i) else {
                                break;
                            };
                            if let Some(program) = load(id, source) {
                                mine.push((i, f(&program)));
                            }
                        }
                        mine
                    })
                    .unwrap()
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    });
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, t)| t).collect()
}

/// Worker threads per corpus pass; `.config/nextest.toml` reserves as many slots.
pub const THREADS: usize = 8;

/// Stack of each corpus worker thread.
const STACK: usize = 64 << 20;

fn load(id: &str, source: &[u8]) -> Option<Program> {
    let checked = [true, false]
        .into_iter()
        .find_map(|engine| check::check(front::parse(source).ok()?, engine).ok())?;
    invar::ownership::check(&checked).ok()?;
    let mir = mir::lower(&checked).ok()?;
    admission::analyze(source, &mir).ok()?;
    Some(Program {
        id: id.to_owned(),
        source: source.to_vec(),
        checked,
        mir,
    })
}
