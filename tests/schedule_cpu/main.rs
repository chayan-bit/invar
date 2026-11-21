#[path = "../../test-support/corpus_sample.rs"]
mod corpus_sample;
#[path = "../../test-support/split_mix.rs"]
mod split_mix;

impl split_mix::Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

mod admission_records;
mod affine_recurrence_scan;
mod checked_fold_carry;
mod cpu_scheduler_fallback;
mod cpu_workers;
mod extremum_intrinsics;
mod fused_map_extraction;
mod lawful_reductions_and_scans;
mod recognition_corpus;
mod select_fold_records;
