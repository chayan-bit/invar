## 6. Units and frames

### 6.1 Quantities and dimension algebra

[N6.1.1] `Quantity<T, Unit>` carries a unit in its type, built by `quantity::<unit>(x)`, and the unit is erased before MIR so generated code equals the unit-free program. `tests/language/units_and_frames.rs::units_and_frames_quantity_erasure_is_byte_identical`
[N6.1.2] The known units are `one m mm um kg g s ms us ns A K mol cd rad urad deg Hz N Nm px`, with dimension product, quotient and integer power in the type. `src/check/unit_math.rs`
[N6.1.3] `+`, `-` and comparison need identical units, `*` and `/` combine dimensions, and `1 / s` is `Hz`. `tests/language/units_and_frames.rs::units_and_frames_dimension_products_quotients_and_rates`
[N6.1.4] `to::<unit>(q)` is the only conversion between scales, there is no implicit conversion, and a unit is not a value. `tests/language/units_and_frames.rs::units_and_frames_quantity_source_errors`
[N6.1.5] A scale conversion that is not exact traps with `UNIT_CONVERSION_INEXACT`. `tests/diagnostics/machine_diagnostics/indexed_and_versions.rs::machine_diagnostics_conversion_trap_is_versioned`
[N6.1.6] `rad` and `deg` conversions use a certified bound on pi and round to nearest even. `src/angle.rs`
[N6.1.7] `px` is a nominal camera unit distinct from `one` and every SI unit. `spec/grammar.ebnf`

```invar outcome=value:1
fn main() -> i32 {
    let force: Quantity<i32, N> = quantity::<N>(2i32);
    let arm: Quantity<i32, m> = quantity::<m>(3i32);
    let torque: Quantity<i32, Nm> = force * arm;
    let second: Quantity<i32, s> = quantity::<s>(1i32);
    let frequency: Quantity<i32, Hz> = 1i32 / second;
    if torque == quantity::<Nm>(6i32) and frequency == 1i32 / second { return 1i32; }
    return 0i32;
}
```

```invar outcome=diag:UNIT_MISMATCH
fn main() -> i32 {
    let x: Quantity<i32, mm> = quantity::<mm>(1i32);
    let y: Quantity<i32, m> = quantity::<m>(1i32);
    let z: Quantity<i32, mm> = x + y;
    return 0i32;
}
```

```invar outcome=trap:UNIT_CONVERSION_INEXACT
fn main() -> i32 {
    let mm: Quantity<i32, mm> = quantity::<mm>(1i32);
    let metres: Quantity<i32, m> = to::<m>(mm);
    return 0i32;
}
```

### 6.2 Frames and transforms

[N6.2.1] `frame Name;` declares a root and `frame Name: Parent;` a child, and the tree needs exactly one root, no cycle, no duplicate and no unknown parent. `tests/language/units_and_frames.rs::units_and_frames_nominal_tree_public_check`
[N6.2.2] `Vector3<T,unit,frame>`, `Point<T,frame>` and `Transform<From,To>` are nominal and are made by `vector3`, `point` and `transform`. `tests/language/units_and_frames.rs::units_and_frames_transform_chain_agrees`
[N6.2.3] `Transform<A,B> * Transform<B,C>` is `Transform<A,C>`, any other composition is `FRAME_MISMATCH`, and a transform also applies to a point or vector in its source frame. `tests/language/units_and_frames.rs::units_and_frames_transform_chain_agrees`
[N6.2.4] A transform rotation is a constant proper signed permutation, and Point and translation coordinates are integer micrometres. `spec/grammar.ebnf`
[N6.2.5] There are at most 128 frames. `src/check/frames.rs::MAX_FRAMES`

```invar outcome=value:1
frame world;
frame base: world;
frame tool: base;

fn main() -> i32 {
    let v: Vector3<i32,um,world> = vector3::<um,world>(1000, 0, 0);
    let p: Point<i32,world> = point::<world>(1000, 0, 0);
    let a: Transform<world,base> = transform::<world,base>(1, 2, 3, 1,0,0, 0,1,0, 0,0,1);
    let b: Transform<base,tool> = transform::<base,tool>(0, 0, 0, 1,0,0, 0,1,0, 0,0,1);
    let composed: Transform<world,tool> = a * b;
    let target: Point<i32,tool> = composed * p;
    let shift: Vector3<i32,um,tool> = composed * v;
    let moved: Point<i32,tool> = target + shift;
    if moved[0u32] == quantity::<um>(2001) { return 1; }
    return 0;
}
```

```invar outcome=diag:FRAME_MISMATCH
frame world;
frame base: world;

fn main() -> i32 {
    let a: Vector3<i32,m,world> = vector3::<m,world>(1, 0, 0);
    let b: Vector3<i32,m,base> = vector3::<m,base>(1, 0, 0);
    let c: Vector3<i32,m,world> = a + b;
    return 0;
}
```

```invar outcome=diag:FRAME_CYCLE
frame world;
frame a: b;
frame b: a;
fn main() -> i32 { return 0i32; }
```

### 6.3 Robotics vocabulary

[N6.3.1] `std::robotics` is the shipped qualified module of generic record types for poses, twists, wrenches, joint vectors, stamps and camera models, with C layouts and ROS mappings. `tests/robotics/robotics_vocabulary.rs::robotics_vocabulary_standard_module_checks`
[N6.3.2] `f32` and `f64` are accepted as wire types of ports. `spec/grammar.ebnf`
[N6.3.3] Camera intrinsics use pixel units, and a focal length in metres is `UNIT_MISMATCH`. `tests/robotics/robotics_vocabulary.rs::robotics_vocabulary_pixel_is_nominal`
[N6.3.4] Joint-vector types carry their count, and a count mismatch is rejected. `tests/robotics/robotics_vocabulary.rs::robotics_vocabulary_rejects_mismatched_units_and_joint_counts`

