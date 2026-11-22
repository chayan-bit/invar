## 7. Bytes, strings and the C ABI

### 7.1 Bytes and String

[N7.1.1] `Bytes` and `String` are distinct immutable owned values of at most 1,048,576 bytes, `len` is the byte length, and each live byte counts as one element toward the ceiling of N12.1.1. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_large_bounded_value` `src/eval.rs::record_cleanup`
[N7.1.2] `b"..."` allows `\xNN` escapes and arbitrary bytes, `"..."` must be valid UTF-8 without `\xNN`, and there is no normalization or implicit conversion. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_utf8_and_byte_values_agree`
[N7.1.3] `Bytes` indexing yields a `u32`, an out-of-range index traps with `INDEX_OUT_OF_BOUNDS`, and `String` has no indexing. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_public_failures`

```invar outcome=value:7
fn main() -> i32 {
    let x: Bytes = b"\xFF";
    let s: String = "é";
    if x[0u32] == 255u32 and len(s) == 2u32 { return 7; }
    return 0;
}
```

```invar outcome=trap:INDEX_OUT_OF_BOUNDS
fn main() -> i32 {
    let bytes: Bytes = b"a";
    if bytes[1u32] == 0u32 { return 1; }
    return 0;
}
```

### 7.2 UTF-8 conversions

[N7.2.1] `bytes_slice(owner, a, b)` copies a range into `Bytes`, `utf8_slice` returns `Result<String,Utf8Error>`, `utf8_decode` validates and `utf8_encode` copies. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_utf8_and_byte_values_agree`
[N7.2.2] These borrow a bare named owner, a reversed or out-of-range range traps, and malformed UTF-8 or a split code point is an ordinary `Err(Utf8Error)` value. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_public_failures`

```invar outcome=value:1
fn main() -> i32 {
    let raw: Bytes = b"\xFF";
    let result = utf8_decode(raw);
    return match move result { Result::Ok(text) => 0i32, Result::Err(error) => 1i32 };
}
```

### 7.3 Foreign C declarations

[N7.3.1] A top-level `extern c fn symbol(args) -> Result<T,ForeignError> oracle name;` declares at most 8 arguments from `i32`, `u32`, `read` or `take` `Bytes` and `String`. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_unsupported_signatures_reject_publicly`
[N7.3.2] The named oracle is an ordinary Invar function with the exact signature, and `check` never loads a library. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_foreign_oracle_declares_and_checks`
[N7.3.3] Reference and MIR execution run the oracle, and native execution needs an explicit absolute `--c-library`. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_foreign_buffers_and_failures_agree`
[N7.3.4] A `take` argument is consumed even when C reports an error, so a later use is `OWN_USE_AFTER_MOVE`. `tests/fixtures/l1_04/foreign-use-after-take.inv`
[N7.3.5] Callbacks, borrowed returns, mutable buffers, C strings and general FFI are rejected. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_unsupported_signatures_reject_publicly`
[N7.3.6] Foreign code is outside the safe-language claims, and the C ABI does not make native dependencies safe. `docs/architecture.md`
[N7.3.7] Native execution loads only the library named by `--c-library`, never a source-relative or current-directory path, and `build --target native` records that path without loading it. `src/native_command.rs`

```invar outcome=value:7 foreign
extern c fn c_add(left: i32, right: i32) -> Result<i32, ForeignError> oracle add_oracle;

fn add_oracle(left: i32, right: i32) -> Result<i32, ForeignError> {
    return Result::Ok(left + right);
}

fn main() -> i32 {
    let answer: Result<i32, ForeignError> = c_add(3i32, 4i32);
    return match move answer { Result::Ok(value) => value, Result::Err(error) => 0i32 };
}
```

### 7.4 The C ABI shim

[N7.4.1] The C shim is `int32_t symbol(const inv_arg_v1 *args, uint32_t argc, inv_result_v1 *out)` with tags, modes and 64-bit lengths frozen in `include/invar_foreign_v1.h`. `include/invar_foreign_v1.h`
[N7.4.2] The library must export `inv_abi_version_v1() == 1`, and a wrong version or missing symbol becomes `Err(ForeignError::Failure())`. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_detectable_foreign_failures_are_typed`
[N7.4.3] Buffer arguments are valid only during the call, and a C-owned result buffer must supply a release callback that runs exactly once after validation and copy. `tests/fixtures/l1_04/foreign.c`
[N7.4.4] A result buffer over 1 MiB, bad tag or version, a null with nonzero length or invalid UTF-8 is a typed error and never truncates. `tests/language/bytes_strings_and_c_abi.rs::bytes_strings_and_c_abi_detectable_foreign_failures_are_typed`

