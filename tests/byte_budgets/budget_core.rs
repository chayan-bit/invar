//! #57 ENG-08 part A: the single `ByteBudget` mechanism. Every test asserts that it ran a nonzero
//! number of cases so an emptied table cannot pass.
use invar::array_limits::{MAX_DEVICE_BYTES, MAX_LIVE_ELEMENTS};
use invar::budget::{
    BudgetExceeded, ByteBudget, DEFAULT_ANALYSIS_BYTES, Footprint, Resource, array_bytes,
};

#[test]
fn default_budget_derives_from_language_limits() {
    let b = ByteBudget::default();
    assert_eq!(b, ByteBudget::standard());
    assert_eq!(b.host_bytes, MAX_LIVE_ELEMENTS * 4);
    assert_eq!(b.device_bytes, MAX_DEVICE_BYTES);
    assert_eq!(b.analysis_bytes, 256 * 1024 * 1024);
    assert_eq!(b.analysis_bytes, DEFAULT_ANALYSIS_BYTES);
    let mut checked = 0;
    for r in Resource::ALL {
        assert!(b.available(r) > 0, "{r:?}");
        checked += 1;
    }
    assert_eq!(checked, 3);
}

#[test]
fn run_budget_and_live_device_only_tighten() {
    let run = ByteBudget::from_run_budget(1 << 20);
    assert_eq!(run.host_bytes, 1 << 20);
    assert_eq!(run.device_bytes, MAX_DEVICE_BYTES);
    assert_eq!(run.analysis_bytes, DEFAULT_ANALYSIS_BYTES);
    let live = run.with_live_device_bytes(4096);
    assert_eq!(live.device_bytes, 4096);
    // A live query above the ceiling never widens it.
    assert_eq!(
        run.with_live_device_bytes(usize::MAX).device_bytes,
        MAX_DEVICE_BYTES
    );
    let wide = ByteBudget::new(usize::MAX, usize::MAX, usize::MAX);
    assert_eq!(wide.capped_by(run), run);
    assert_eq!(run.capped_by(wide), run);
    let tight = ByteBudget::new(1, 2, 3);
    assert_eq!(run.capped_by(tight), tight);
    assert_eq!(
        ByteBudget::ZERO
            .with_host_bytes(5)
            .with_device_bytes(6)
            .with_analysis_bytes(7),
        ByteBudget::new(5, 6, 7)
    );
}

#[test]
fn admits_at_exactly_the_footprint_and_refuses_one_byte_below() {
    let mut cases = 0;
    for (host, device, analysis) in [
        (1usize, 0usize, 0usize),
        (4100, 0, 64),
        (4100, 900, 64),
        (7, 8, 9),
    ] {
        let need = Footprint::ZERO
            .host(Some(host))
            .device(Some(device))
            .analysis(Some(analysis));
        let exact = ByteBudget::new(host, device, analysis);
        assert_eq!(exact.admit("indexed", &need), Ok(()));
        for resource in Resource::ALL {
            let n = need.bytes(resource).unwrap();
            if n == 0 {
                continue;
            }
            let mut below = exact;
            match resource {
                Resource::Host => below.host_bytes -= 1,
                Resource::Device => below.device_bytes -= 1,
                Resource::Analysis => below.analysis_bytes -= 1,
            }
            let refusal = below.admit("indexed", &need).unwrap_err();
            assert_eq!(
                refusal,
                BudgetExceeded {
                    family: "indexed",
                    resource,
                    needed: Some(n),
                    available: n - 1,
                }
            );
            assert!(refusal.to_string().starts_with("BUDGET_EXCEEDED {"));
            cases += 1;
        }
    }
    assert!(cases >= 8, "{cases}");
}

#[test]
fn zero_budget_refuses_every_device_region_and_no_empty_one() {
    let mut refused = 0;
    for bytes in [1usize, 4, 4096, 1 << 30] {
        let device = Footprint::ZERO.device(Some(bytes));
        let err = ByteBudget::ZERO.admit("gather", &device).unwrap_err();
        assert_eq!(err.resource, Resource::Device);
        assert_eq!(err.available, 0);
        // The same region is admitted once the device ceiling covers it.
        assert!(
            ByteBudget::ZERO
                .with_device_bytes(bytes)
                .admit("gather", &device)
                .is_ok()
        );
        refused += 1;
    }
    assert_eq!(refused, 4);
    // A region that needs no bytes of anything is not refused by a zero budget.
    assert_eq!(ByteBudget::ZERO.admit("gather", &Footprint::ZERO), Ok(()));
}

#[test]
fn overflow_is_a_refusal_not_a_panic() {
    assert_eq!(array_bytes::<u32>(usize::MAX), None);
    assert_eq!(array_bytes::<u64>(usize::MAX / 8 + 1), None);
    assert_eq!(array_bytes::<u32>(1000), Some(4000));
    let mut cases = 0;
    for resource in Resource::ALL {
        let overflowed = Footprint::ZERO
            .charge(resource, Some(usize::MAX))
            .charge(resource, Some(1));
        assert_eq!(overflowed.bytes(resource), None);
        // Overflow stays sticky: later charges cannot launder it.
        assert_eq!(overflowed.charge(resource, Some(0)).bytes(resource), None);
        let err = ByteBudget::new(usize::MAX, usize::MAX, usize::MAX)
            .admit("frontier", &overflowed)
            .unwrap_err();
        assert_eq!(
            (err.resource, err.needed, err.available),
            (resource, None, usize::MAX)
        );
        assert!(err.to_string().contains("needed: overflow"));
        let poisoned = Footprint::ZERO.charge(resource, array_bytes::<u64>(usize::MAX));
        assert_eq!(poisoned.bytes(resource), None);
        cases += 1;
    }
    assert_eq!(cases, 3);
    let sum = Footprint::ZERO
        .host(Some(usize::MAX))
        .plus(Footprint::ZERO.host(Some(1)).device(Some(5)));
    assert_eq!(sum.bytes(Resource::Host), None);
    assert_eq!(sum.bytes(Resource::Device), Some(5));
}

#[test]
fn footprints_add_per_resource_for_running_totals() {
    let mut total = Footprint::ZERO;
    let step = Footprint::ZERO.host(Some(100)).analysis(Some(10));
    let budget = ByteBudget::new(350, 0, 1000);
    let mut admitted = 0;
    loop {
        let next = total.plus(step);
        if budget.admit("vec", &next).is_err() {
            break;
        }
        total = next;
        admitted += 1;
    }
    assert_eq!(admitted, 3);
    assert_eq!(total.bytes(Resource::Host), Some(300));
    assert_eq!(total.bytes(Resource::Analysis), Some(30));
}

#[test]
fn refusal_text_is_stable() {
    let e = BudgetExceeded {
        family: "histogram",
        resource: Resource::Analysis,
        needed: Some(9),
        available: 8,
    };
    assert_eq!(
        e.to_string(),
        "BUDGET_EXCEEDED { family: histogram, resource: analysis, needed: 9, available: 8 }"
    );
}
