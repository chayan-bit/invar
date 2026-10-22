# S1-09 - Integration target: Nix modules and signed image manifest

Release: v0.3 (milestone S1); planning contract, to be expanded into a full contract before implementation.

Dependencies: S1-08, R2-04.

## Goal

Generate the Nix modules and the signed image manifest for a product from the checked system, so the deployed image is a deterministic output of the same source.

## Motivation

The image is built with Nix and updated with RAUC A/B and TUF metadata; generating them from the system IR removes the last hand-maintained deployment glue.

## Scope

- Generate NixOS modules (services per component with declared isolation, resource limits from `resource` clauses as systemd limits, PREEMPT_RT kernel selection) and a flake output for the product image.
- Generate a RAUC A/B bundle manifest and TUF targets metadata with the product's compatibility manifest; signing keys come from an external secret store and never enter the repository.
- Product-line variants from R2-04 select which components and parameters enter each image.

## Acceptance sketch

- Integration test binary `issue_s1_09_nix_image_target`: deterministic generation, a built image for the S1-06 cell in a VM test, and a signature verification test with a throwaway test key.

## Out of scope

Jetson-specific images (R1-07 owns jetpack-nixos use), OTA server infrastructure, Yocto.
