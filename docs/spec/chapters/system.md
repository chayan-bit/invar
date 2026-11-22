## 8. System dialect

### 8.1 Declarations

[N8.1.1] A system description is ordinary Invar plus `component`, `extern component`, `connect`, `flow`, `frame`, `product` and `system`. `docs/system-integration-v1.md`
[N8.1.2] A `component` has `port`, `param`, `resource`, `license`, `isolation`, `wcet`, `deadline` and `impl` clauses. `spec/grammar.ebnf`
[N8.1.3] A port is `port in|out name: Type @ RATE [latency L] [layout c "name"] [bus name] [optional];`, and a kernel signature and a port type are the same type. `spec/grammar.ebnf`
[N8.1.4] `connect a.p -> b.q policy P;` joins an output to an input where the policy is `sample-and-hold`, `queue(n)` or `decimate(n)`. `docs/system-integration-v1.md`
[N8.1.5] Constants are exact decimal literals with units, evaluated as exact rationals and serialized as canonical decimal strings. `docs/system-integration-v1.md`
[N8.1.6] A source needs exactly one `product` and one `system`, and an instance is `name: Component;` inside the system. `tests/fixtures/s1_01/negative/S1005-no-product.inv`
[N8.1.7] An optional input or an unconnected output is the warning `S1011`, and other well-formedness failures are `S1000` to `S1010` errors. `tests/robotics/system_dialect.rs::system_dialect_optional_input_and_output_warning`
[N8.1.8] `connect` requires equal type, unit and frame at both ends, else `S1002`. `tests/robotics/system_dialect.rs::system_dialect_negative_fixtures_have_registered_spans`
[N8.1.9] A system has no runtime semantics, and it is evaluated at compile time within the usual fuel and size limits. `docs/system-integration-v1.md`

```invar outcome=system:ok
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

product Bench {
    distribution internal;
    commercial false;
    robot menagerie::franka_emika_panda;
    capacity cpu_cores = 4;
}

system cell for Bench { sensor: Sensor; logger: Logger; }
```

```invar outcome=system:S1002
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
    port in reading: Quantity<i32,mm> @ 10 Hz;
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

### 8.2 The invar-system IR

[N8.2.1] A checked system is serialized as canonical JSON named `invar-system-v12`, byte-identical for identical input, and validated by `schema/invar-system-v12.schema.json`. `tests/robotics/system_dialect.rs::system_dialect_emits_checked_canonical_ir`
[N8.2.2] Every source emits v12, and the `robots` and `cells` arrays appear only when the source declares a robot or cell model. `src/system/compiler.rs::models`
[N8.2.3] An unknown version is rejected, and relabelling a document does not make it valid. `tests/robotics/graph_checks.rs::graph_checks_twelve_component_cell_and_golden_report`
[N8.2.4] Permuting module order of the same system gives byte-identical IR. `tests/robotics/system_dialect.rs::system_dialect_module_permutation_is_byte_identical`
[N8.2.5] Deserializing an IR does not establish a source check, and only a source-compiled model can pass the graph checks. `tests/robotics/graph_checks.rs::graph_checks_twelve_component_cell_and_golden_report`

### 8.3 Generators

[N8.3.1] Every generator is a pure function of the checked IR with byte-identical output, and it reads no source text, environment, clock or network. `docs/system-integration-v1.md`
[N8.3.2] `invar system gen ros2|mjcf|urdf PATH --out NEW_DIRECTORY` writes a new directory and refuses an existing one. `tests/robotics/ros2_target.rs::ros2_target_public_errors_leave_no_output`
[N8.3.3] The ROS 2 target needs a graph-checked system, else `S1500`, and its output is deterministic. `tests/robotics/ros2_target.rs::ros2_target_public_generation_is_deterministic`
[N8.3.4] A system with a robot and cell model also generates MJCF and, where it can be preserved, URDF, and unsupported URDF features are `S1607`. `docs/system-integration-v1.md`
[N8.3.5] A generated ROS 2 workspace is mock-hardware accepted on ARM64 only, and a live simulation run is not claimed. `README.md`

