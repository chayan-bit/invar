# laws_synth

Proof obligations for law certificates: law formulas, their SMT-LIB2 text and an in-repo decision procedure that discharges them, so no external solver is trusted.

## Where to start

- [formula.rs](formula.rs) defines quantifier-free law formulas; [prove.rs](prove.rs) is the decision procedure (`Verdict`, `Mutation`); [linear.rs](linear.rs) does Fourier-Motzkin elimination for case splits.
- [certificate.rs](certificate.rs) defines lifting certificates, files committed beside the source (`LawCertificate`).

## Position in the pipeline

Used by the normal form in [general_net](../general_net/README.md).

## Tests

- `cargo nextest run --locked --test laws_prover` (`certificate`, `lift`, `lifting_laws`, `prove`).

## Rules to keep

- The prover must reject every mutation it is given; a verdict is never inferred from a name.

## Read next

- [net normal form](../../docs/net-normal-form-v01.md)
