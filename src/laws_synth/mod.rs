//! Law proof obligations for #281 (ENG-02): the formulas a law certificate
//! states, their SMT-LIB2 text, and the in-repo decision procedure that
//! discharges them (no external solver is trusted; [`prove`] states the
//! trust root, its fragment and why each rule is sound). A synthesizer only
//! proposes; the checker re-derives a certificate's formula from the net and
//! re-decides it with [`prove::decide`].

pub mod formula;
mod linear;
pub mod prove;

/// Solver id and version recorded in law certificates.
pub const SOLVER: &str = "invar-prove";
pub const SOLVER_VERSION: u32 = 1;
