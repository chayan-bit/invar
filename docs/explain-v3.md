# Explicit sort explanation v3

Sort reports now use `invar-explain-v3`, schema_version 3 and device schema
`invar-explain-device-v3`. The frozen v1/v2 schemas remain unchanged.
Each checked `explicit_sorts` site adds a required `calibration_id` with the
C14 family spelling `sort.<op>.<type>`, such as `sort.argsort.f32` or
`sort.sort_by_key.u32`. Operations remain overloaded by their key array type.
These ids identify calibration families; they do not assert measured costs.

`sort-cost-unmeasured` is registered in AI-01's `explain_reason_codes` in
`schema/diagnostic-codes-v1.json`. Automatic execution still selects CPU with
no cost estimate. Existing implementation receipts, source identity and
failure semantics are unchanged from v2. Non-sort reports retain v1.
