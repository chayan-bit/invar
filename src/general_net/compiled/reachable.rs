//! Conservative CFG/call closure from the selected entry. Both branch arms
//! remain reachable regardless of literals or inputs. This removes emitted
//! code only; whole-candidate admission and source observations stay intact.
use super::{Error, mir};
use std::collections::BTreeSet;

pub(super) fn blocks(
    program: &mir::Program,
    entry: usize,
) -> Result<BTreeSet<(usize, usize)>, Error> {
    let mut seen = BTreeSet::new();
    let mut pending = vec![(
        entry,
        program.functions.get(entry).ok_or(Error::Entry)?.entry,
    )];
    while let Some((f, b)) = pending.pop() {
        if !seen.insert((f, b)) {
            continue;
        }
        let function = program.functions.get(f).ok_or(Error::State)?;
        // Foreign stubs have no CFG. Whole-candidate admission below still
        // returns the original typed unsupported error, in its original order.
        if function.foreign_oracle.is_some() {
            continue;
        }
        let block = program
            .functions
            .get(f)
            .and_then(|f| f.blocks.get(b))
            .ok_or(Error::State)?;
        for instruction in &block.instructions {
            if let mir::Operation::Call { name, .. } = &instruction.kind
                && let Some((callee, function)) = program
                    .functions
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == *name)
            {
                pending.push((callee, function.entry));
            }
        }
        match block.end.as_ref().ok_or(Error::State)? {
            mir::Terminator::Goto { target, .. } => pending.push((f, *target)),
            mir::Terminator::Branch { yes, no, .. } => {
                pending.push((f, *yes));
                pending.push((f, *no));
            }
            mir::Terminator::Return { .. } => {}
        }
    }
    Ok(seen)
}

#[cfg(test)]
mod tests {
    #[test]
    fn pruned_owned_numeric_cuda_continuation_compiles() {
        let source = b"fn run(a:f32,n:u32)->f32{var xs:Array<f32> = filled_f32(n+1u32,a);xs[0u32]=xs[0u32]*a+a;return xs[0u32];}fn main()->i32{return 0;}";
        let artifact = crate::general_net::provenance::compile_resumable_owned_numeric(
            source,
            "run",
            super::super::Target::Cuda,
            Default::default(),
            super::super::OwnedLimits {
                elements: 64,
                allocations: 16,
                cleanup_events: 32,
            },
        )
        .unwrap();
        assert_eq!(artifact.artifact().abi_version(), 12);
        assert!(artifact.artifact().owned_numeric_cuda_facts().is_ok());
    }

    /// The L1-04 foreign-oracle program, inline: compiler sources must not name
    /// fixture paths (`corpus_compiler_sources_never_name_the_corpus`).
    const FOREIGN_ORACLE: &str = "\
extern c fn c_add(left: i32, right: i32) -> Result<i32, ForeignError> oracle add_oracle;

fn add_oracle(left: i32, right: i32) -> Result<i32, ForeignError> {
  return Result::Ok(left + right);
}

fn main() -> i32 {
  let answer: Result<i32, ForeignError> = c_add(3i32, 4i32);
  return match move answer { Result::Ok(value) => value, Result::Err(error) => 0i32 };
}
";

    #[test]
    fn foreign_stubs_keep_whole_candidate_typed_admission() {
        let checked = crate::check::check(
            crate::front::parse(FOREIGN_ORACLE.as_bytes()).unwrap(),
            true,
        )
        .unwrap();
        let mir = crate::mir::lower(&checked).unwrap();
        let result =
            super::super::Program::new(&mir, "main", Default::default(), None, false, None, None);
        assert!(
            matches!(result, Err(super::Error::UnsupportedType)),
            "wrong refusal: {:?}",
            result.err()
        );
    }
}
