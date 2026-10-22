# Explicit sort explanation v2

Historical reports containing explicit sort sites use `schema: invar-explain-v2`,
`schema_version: 2`, and, when executed, `device_schema: invar-explain-device-v2`.
The frozen v1 schemas and reports for other source programs are unchanged.
The strict schemas are in `schema/explain-v2.schema.json` and
`schema/invar-explain-device-v2.schema.json`.

`explicit_sorts` lists checked call sites, spans, operations, the conservative
CPU default, and rule `explicit-stable-sort-v1`. Static sites do not establish
execution. Reached execution choices carry an `implementation` naming serial
or joined CPU radix, CUDA radix, Metal radix, or an empty host no-op.
Automatic sort choices use `sort-cost-unmeasured` and no cost estimate;
a generic loop bandwidth model does not calibrate radix sorting. Device sort
modules/pipelines are compiled per call and do not claim cache hits.
The existing source identity, region explanation and execution/failure
semantics remain as documented for v1. Ordinary run diagnostics retain v1.

C14 typed calibration ids are added in [v3](explain-v3.md); new sort reports use v3.
