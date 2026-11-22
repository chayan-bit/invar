## 9. Graph checks

[N9.0.1] Graph checks run on every source-compiled system. `tests/robotics/graph_checks.rs::graph_checks_twelve_component_cell_and_golden_report`
[N9.0.2] Diagnostics are ordered by family and then span, and the same input gives the same output. `tests/robotics/graph_checks.rs::graph_checks_diagnostic_families_keep_stable_order`
[N9.0.3] The checker proves structure only, and deadlines and WCET are declared budgets and not measurements. `docs/system-integration-v1.md`

### 9.1 Rates and queues

[N9.1.1] Under `queue(n)` the producer rate may not exceed the consumer rate, else `S1100`. `src/system/graph.rs`
[N9.1.2] A component deadline may not exceed the period of its ports, else `S1101`, and a WCET above its deadline is `S1101`. `tests/fixtures/s1_03/negative/S1101-wcet-over-deadline.inv`
[N9.1.3] `decimate(n)` needs an integral producer to consumer ratio, else `S1100`. `tests/fixtures/s1_03/negative/S1100-decimate-ratio.inv`

```invar outcome=system:S1101
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 200 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.2 Latency, flows and budgets

[N9.2.1] A flow `flow f: a.p -> b.q within T;` sums the WCET of each distinct instance on it, each hop's source-port latency, and the consumer period for `sample-and-hold` and `decimate` hops, and the sum may not exceed `T`, else `S1102`. `src/system/graph.rs`
[N9.2.2] A flow hop that is not a connection is `S1102`. `tests/fixtures/s1_03/negative/S1102-disconnected-flow.inv`

```invar outcome=system:S1102
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);
flow loop: sensor.reading -> logger.reading within 1 ms;

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.3 Frame-tree closure

[N9.3.1] Every frame used by a port or transform must be reachable from exactly one root, and an unreachable frame or a second root is `S1200`. `tests/fixtures/s1_03/negative/S1200-unreachable-frame.inv`
[N9.3.2] Every tree edge used by a transform needs one declared transform source, else `S1201`, and two sources are `S1202`. `tests/fixtures/s1_03/negative/S1201-missing-transform-source.inv`

### 9.4 Resources

[N9.4.1] The sum of each instance's `cpu_cores`, `memory`, `gpu_memory` and per-bus bandwidth may not exceed the product `capacity`, else `S1250`. `tests/fixtures/s1_03/negative/S1250-cpu-over.inv`
[N9.4.2] Bus bandwidth is declared port rate times canonical C layout bytes times 8, and it excludes serialization and network overhead. `src/system.rs::report`
[N9.4.3] A bus not declared by the product is `S1251`, and a port type with no canonical layout is `S1252`. `tests/fixtures/s1_03/negative/S1251-undeclared-bus.inv`

```invar outcome=system:S1250
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 2;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 1;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

### 9.5 License and isolation

[N9.5.1] Each component carries an SPDX expression and one isolation of `in-process`, `separate-process` or `separate-partition`, and an unknown or unparsable license is rejected. `tests/fixtures/s1_03/negative/S1305-unknown-no-grant.inv`
[N9.5.2] For `distribution proprietary-binary`, GPL must be a separate unmodified process and AGPL also may not be network exposed. `docs/system-integration-v1.md`
[N9.5.3] LGPL in process must be dynamic and replaceable, MPL-2.0 in process must be unmodified, and non-commercial terms are rejected when `commercial true`. `docs/system-integration-v1.md`
[N9.5.4] Proprietary or unknown third-party terms need a declared grant id, and these checks are an engineering check and not legal advice. `docs/system-integration-v1.md`
[N9.5.5] The v0.1 check covers components, and the check over the Nix image closure is not part of v0.1. `docs/system-integration-v1.md`

```invar outcome=system:S1300
frame world;

component Sensor {
    port out reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "MIT" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar read_sensor;
}

component Logger {
    port in reading: Quantity<i32,m> @ 10 Hz;
    resource cpu_cores = 1;
    license "GPL-3.0-only" modified false;
    isolation in-process;
    wcet 1 ms;
    deadline 100 ms;
    impl invar log_reading;
}

fn read_sensor() -> i32 { return 0; }
fn log_reading() -> i32 { return 0; }

connect sensor.reading -> logger.reading policy queue(4);

product Bench {
    distribution proprietary-binary;
    commercial true;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

