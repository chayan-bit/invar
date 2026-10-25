#![forbid(unsafe_code)]

#[path = "../support/law_bench.rs"]
mod bench;

#[path = "../support/law_collective.rs"]
mod support;

fn verdict(ratio: f64, u95: f64) -> &'static str {
    if ratio > 1.0 {
        "FAIL"
    } else if u95 > 1.05 {
        "INCONCLUSIVE"
    } else {
        "PASS"
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.0
}

// Frozen before hardware timing: ratio of medians, hierarchical paired
// bootstrap (sessions, then paired blocks), 2000 resamples, seed 0x8d5.
fn stats(sessions: &[Vec<[f64; 4]>], scope: usize) -> (f64, f64, f64, f64) {
    let ratio = |rows: &[[f64; 4]]| {
        let mut g = rows.iter().map(|r| r[scope]).collect::<Vec<_>>();
        let mut b = rows.iter().map(|r| r[scope + 1]).collect::<Vec<_>>();
        let (g, b) = (median(&mut g), median(&mut b));
        (g / b, g, b)
    };
    assert!(
        !sessions.is_empty()
            && sessions
                .iter()
                .all(|s| !s.is_empty() && s.iter().flatten().all(|x| x.is_finite() && *x > 0.0))
    );
    let all = sessions.iter().flatten().copied().collect::<Vec<_>>();
    let (point, g, b) = ratio(&all);
    let mut rng = support::Rng(0x8d5);
    let mut boot = Vec::new();
    for _ in 0..2000 {
        let mut sample = Vec::new();
        for _ in sessions {
            let s = &sessions[rng.next() as usize % sessions.len()];
            for _ in s {
                sample.push(s[rng.next() as usize % s.len()]);
            }
        }
        boot.push(ratio(&sample).0);
    }
    boot.sort_by(f64::total_cmp);
    (point, boot[1900], g, b)
}

#[test]
fn p8_native_fold_matches_oracle_representation() {
    for kind in 0..2 {
        for n in [1, 2, 1023, 4097] {
            let (input, expected) = bench::data(kind, n);
            assert_eq!(bench::native(kind, &input), expected);
        }
    }
    assert_eq!(bench::native(0, &[]), vec![0; 6]);
    assert_eq!(bench::native(1, &[0, 7, 1, 5, 0, 3]), vec![1, 5]);
}

#[test]
fn p8_cost_model_refuses_tied_host_reductions_to_native() {
    use invar::general_net::law_collective::{
        Calibration, Residency, Route,
        cost::{Mutation, route_with},
        route,
    };
    // A staging link far below device bandwidth and a fast native fold.
    let calibration = Calibration::from_json(&calibration_json(8.0e8, 6.0e6, 1.2e7, 1.0)).unwrap();
    let (sum, scan) = (bench::repr(0), bench::repr(2));
    let tied = |r, s, residency, m| route_with(r, s, 1 << 26, residency, &calibration, m);
    assert!(matches!(
        route(&sum, false, 1 << 26, Residency::Host, &calibration),
        Route::Native(_)
    ));
    assert_eq!(
        route(&sum, false, 1 << 26, Residency::Device, &calibration),
        Route::Collective
    );
    assert_eq!(
        route(&scan, true, 1 << 26, Residency::Host, &calibration),
        Route::Collective
    );
    assert_eq!(
        route(&sum, false, 0, Residency::Host, &calibration),
        Route::Collective
    );
    // A native fold slower than the collective floor keeps the collective.
    let slow = Calibration::from_json(&calibration_json(8.0e8, 6.0e6, 1.0e6, 1.0)).unwrap();
    assert_eq!(
        route(&sum, false, 1 << 26, Residency::Host, &slow),
        Route::Collective
    );
    // Each named mutation flips a decision asserted above.
    assert_eq!(
        tied(&sum, false, Residency::Host, Mutation::DropHostLink),
        Route::Collective
    );
    assert!(matches!(
        tied(
            &sum,
            false,
            Residency::Device,
            Mutation::RefuseDeviceResident
        ),
        Route::Native(_)
    ));
    assert!(matches!(
        tied(&scan, true, Residency::Host, Mutation::RefuseScan),
        Route::Native(_)
    ));
    let mut no_source = calibration_json(1.0, 1.0, 1.0, 1.0);
    no_source["source"] = serde_json::Value::Null;
    let mut zero = calibration_json(1.0, 1.0, 1.0, 1.0);
    zero["passes"][0]["peak_bytes_per_ms"] = serde_json::json!(0.0);
    zero["worst"]["peak_bytes_per_ms"] = serde_json::json!(0.0);
    let mut no_passes = calibration_json(1.0, 1.0, 1.0, 1.0);
    no_passes["passes"] = serde_json::json!([]);
    // The best figure must be the fastest recorded pass, the worst the slowest.
    let mut not_best = calibration_json(1.0, 1.0, 1.0, 1.0);
    not_best["peak_bytes_per_ms"] = serde_json::json!(2.0);
    for bad in [no_source, zero, no_passes, not_best] {
        assert!(Calibration::from_json(&bad).is_err());
    }
}

/// A recorded calibration with two passes: the given figures, and a second
/// pass `slow` times slower in every figure.
fn calibration_json(peak: f64, link: f64, native: f64, slow: f64) -> serde_json::Value {
    let pass = |k: f64| {
        serde_json::json!({
            "peak_bytes_per_ms": peak / k,
            "link_bytes_per_ms": link / k,
            "native_bytes_per_ms": native / k,
        })
    };
    let mut value = pass(1.0);
    value["worst"] = pass(slow);
    value["passes"] = serde_json::json!([pass(1.0), pass(slow)]);
    value["source"] = serde_json::json!("unit-test calibration");
    value
}

#[test]
fn p8_cost_model_marginal_decisions_are_deterministic_per_file() {
    use invar::general_net::law_collective::{
        Calibration, Residency, Route, TIE,
        cost::{Mutation, decide_with},
        decide,
    };
    let sum = bench::repr(0);
    // Infinite peak, so ratio = native / link exactly. The second, slower pass
    // divides the link by `link_slow` and the native fold by `native_slow`.
    let file = |ratio: f64, [link_slow, native_slow]: [f64; 2]| {
        let mut v = calibration_json(1.0e300, 1.0e7, ratio * 1.0e7, 1.0);
        for key in ["link_bytes_per_ms", "native_bytes_per_ms"] {
            let slow = if key == "link_bytes_per_ms" {
                link_slow
            } else {
                native_slow
            };
            let x = v[key].as_f64().unwrap() / slow;
            v["passes"][1][key] = serde_json::json!(x);
            v["worst"][key] = serde_json::json!(x);
        }
        Calibration::from_json(&v).unwrap()
    };
    let run = |c: &Calibration, m| decide_with(&sum, false, 1 << 26, Residency::Host, c, m);
    let native =
        |d: &invar::general_net::law_collective::Decision| matches!(d.route, Route::Native(_));
    // Two files either side of the threshold, both inside the margin.
    let [below, above] = [0.99 * TIE, 1.01 * TIE];
    for (slow, refused) in [([1.0, 1.2], false), ([1.2, 1.0], true)] {
        let [b, a] = [file(below, slow), file(above, slow)];
        let [db, da] = [&b, &a].map(|c| run(c, Mutation::None));
        assert!(db.marginal && da.marginal);
        // The slowest passes decide, so crossing the threshold changes nothing:
        // a busy native pass keeps the collective, a busy link refuses it.
        assert_eq!(db.route, da.route);
        assert_eq!(native(&db), refused);
        // The same file always gives the same decision, re-parsed or not.
        for _ in 0..8 {
            assert_eq!(decide(&sum, false, 1 << 26, Residency::Host, &b), db);
            assert_eq!(run(&file(below, slow), Mutation::None), db);
        }
        // Without the margin, the threshold alone splits the two files.
        let [mb, ma] = [&b, &a].map(|c| run(c, Mutation::IgnoreMargin));
        assert!(!native(&mb) && native(&ma));
    }
    // Outside the margin the best figures decide and nothing is marginal.
    for (ratio, refused) in [(0.85 * TIE, false), (1.15 * TIE, true)] {
        let d = run(&file(ratio, [1.0, 1.2]), Mutation::None);
        assert!(!d.marginal);
        assert_eq!(native(&d), refused);
    }
}

#[test]
fn p8_d5_statistics_preserve_paired_sessions_and_detect_failure() {
    let equal = vec![vec![[2.0, 2.0, 4.0, 4.0]; 30]; 3];
    assert_eq!(stats(&equal, 0), (1.0, 1.0, 2.0, 2.0));
    let slow = vec![vec![[4.0, 2.0, 8.0, 4.0]; 30]; 3];
    assert_eq!(stats(&slow, 2), (2.0, 2.0, 8.0, 4.0));
}

#[test]
#[ignore = "report only; needs the pre-registered saved sessions per device"]
fn p8_d5_report() {
    use serde_json::{Value, json};
    use std::{fs, path::PathBuf};
    let dir = PathBuf::from(std::env::var_os("P8_D5_OUT").unwrap());
    let mut report = String::from(
        "# P8 D5 template measurements\n\nFlat tier: ratio <= 1.00 and one-sided U95 <= 1.05. Pooled warm allocations on both sides. Resident includes scratch reset and all stages. Template AoS inputs exclude source mapping/replay/publication, so these measurements do not establish ordinary-source parity. A `warm (native)` scope is a host-resident reduction the calibrated cost model refused: its generated column is the native host fold. `marginal` marks a route decided by the slowest calibration passes because the best passes put it within 5% of the tie. Metal warm input is caller-owned page-aligned host memory wrapped without a copy on both sides. Halves gives the verdicts of the first and second half of the sessions separately (reported, never used to pass a cell).\n\n| GPU | Collective | n | Scope | Generated ms | Handwritten ms | Ratio | U95 | Result | Halves |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for gpu in ["metal", "cuda"] {
        let count = bench::sessions(gpu == "metal");
        let files = (0..count)
            .map(|s| fs::read_to_string(dir.join(format!("{gpu}-{s}.jsonl"))))
            .collect::<Vec<_>>();
        if files.iter().any(Result::is_err) {
            report.push_str(&format!(
                "\n{gpu}: incomplete independent sessions, no D5 bound.\n\n"
            ));
            continue;
        }
        let records = files
            .into_iter()
            .map(|f| {
                f.unwrap()
                    .lines()
                    .map(|l| serde_json::from_str::<Value>(l).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for kind in 0..4 {
            for power in 20..=26 {
                let n = 1u32 << power;
                let rows = records
                    .iter()
                    .map(|rs| {
                        let r = rs
                            .iter()
                            .find(|r| r["kind"] == bench::name(kind) && r["length"] == n)
                            .unwrap();
                        serde_json::from_value::<Vec<[f64; 4]>>(r["result"]["samples_ms"].clone())
                            .ok()
                    })
                    .collect::<Option<Vec<_>>>();
                let routes = records
                    .iter()
                    .map(|rs| {
                        rs.iter()
                            .find(|r| r["kind"] == bench::name(kind) && r["length"] == n)
                            .map_or("collective", |r| {
                                r["warm_route"].as_str().unwrap_or("collective")
                            })
                    })
                    .collect::<Vec<_>>();
                assert!(
                    routes.iter().all(|r| *r == routes[0]),
                    "sessions disagree on the route"
                );
                let marginal = records.iter().any(|rs| {
                    rs.iter().any(|r| {
                        r["kind"] == bench::name(kind)
                            && r["length"] == n
                            && r["warm_marginal"] == true
                    })
                });
                let warm = match (routes[0] == "native", marginal) {
                    (true, true) => "warm (native, marginal)",
                    (true, false) => "warm (native)",
                    (false, true) => "warm (marginal)",
                    (false, false) => "warm",
                };
                if let Some(rows) = rows {
                    assert!(rows.iter().all(|s| s.len() == 30));
                    for (scope, name) in [(0, "resident"), (2, warm)] {
                        let (r, u, g, b) = stats(&rows, scope);
                        let halves = if count >= 6 {
                            let (a, z) = rows.split_at(count / 2);
                            let [a, z] = [a, z].map(|h| {
                                let (r, u, _, _) = stats(h, scope);
                                verdict(r, u)
                            });
                            format!("{a}/{z}")
                        } else {
                            "-".into()
                        };
                        report.push_str(&format!("| {gpu} | {} | 2^{power} | {name} | {g:.6} | {b:.6} | {r:.4} | {u:.4} | {} | {halves} |\n",bench::name(kind),verdict(r, u)));
                    }
                } else {
                    report.push_str(&format!("| {gpu} | {} | 2^{power} | both | unavailable | unavailable | - | - | OPEN | - |\n",bench::name(kind)));
                }
            }
        }
    }
    fs::write(dir.join("d5-report.md"), &report).unwrap();
    fs::write(dir.join("d5-protocol.json"),serde_json::to_vec_pretty(&json!({"seed":"0x8d5","resamples":2000,"unit":"independent process session, paired blocks within session","estimator":"ratio of medians","warmups":10,"samples":30,"sessions":{"metal":bench::sessions(true),"cuda":bench::sessions(false)},"tier":"flat","limits":[1.0,1.05]})).unwrap()).unwrap();
    println!("{report}");
}
