//! Bounded parallel map with large thread stacks (the reference evaluator
//! recurses on the host stack).
/// Runs `f` over programs on bounded parallel threads with large stacks.
pub fn map<P: Sync, T: Send>(programs: &[P], f: impl Fn(&P) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<T>>> = programs
        .iter()
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn_scoped(scope, || {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(p) = programs.get(i) else { break };
                        *slots[i].lock().unwrap() = Some(f(p));
                    }
                })
                .unwrap();
        }
    });
    slots
        .into_iter()
        .map(|s| s.into_inner().unwrap().unwrap())
        .collect()
}
