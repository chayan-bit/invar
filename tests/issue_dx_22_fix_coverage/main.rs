//! #307 (DX-22): the fix engine and its generators. Every emitted fix is
//! validated by applying it, reparsing and rechecking.

mod cli;
mod common;
mod coverage;
mod engine;
mod hoist;
mod moves;
mod names;
mod ownership;
mod rounds;
