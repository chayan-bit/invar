# S1-08 - License and isolation analysis over the Nix image closure

Release: v0.2 (milestone S1); planning contract, to be expanded into a full contract before implementation.

Dependencies: S1-06.

## Goal

Extend the S1-03 component-level license and isolation check to every derivation in the Nix closure of the product image.

## Motivation

Component licenses do not cover transitive dependencies (a permissive component can pull an LGPL or GPL library); the shipped image is what counts for a proprietary-binary distribution.

## Scope

- Read `meta.license` and linking information for every derivation in the closure of the product's Nix image (from `nix path-info --recursive` and derivation metadata), map it to SPDX, and apply the design note section 4 lattice with the linking mode derived from how each component's binary links it (dynamic, static, separate process).
- Unknown or missing license metadata is a finding, never a pass; an allowlist with justification per entry is committed with the product.
- Results fold into `invar-diagnostics-v1` under `S1300` to `S1399` and into the evidence report.

## Acceptance sketch

- Integration test binary `issue_s1_08_nix_closure_licenses` over a fixture closure with a planted static LGPL library, a GPL in-process library and a missing license, each rejected with its code.
- The S1-06 cell's closure is analysed and its report is committed.

## Out of scope

Legal review, SBOM formats beyond one generated SPDX document, container images not built with Nix.
