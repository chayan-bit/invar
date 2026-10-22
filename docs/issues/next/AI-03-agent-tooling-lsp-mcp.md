# AI-03 - Agent tooling: LSP server, MCP server and fast incremental checking

Release: v0.2 (milestone A1); planning contract, to be expanded into a full contract before implementation.

Dependencies: AI-02.

## Goal

Give AI agents and editors fast, structured access to Invar's types, ownership and plan decisions.

## Motivation

Invar's robotics direction is an OEM supplier model: each new manufacturer's hardware should be cheap to support, because robot-specific facts become typed compiler inputs instead of per-customer hand-written code.

## Scope

- An LSP server (diagnostics, hover types, go to definition, code actions from AI-01 fixes) built on a maintained LSP crate.
- An MCP server exposing types, ownership facts, plan decisions, fallback reasons and the AI-01 explain document.
- Fast incremental checking so that an edit-check loop stays interactive on the example library.

## Acceptance sketch

- LSP and MCP responses validate against the AI-01 schemas.
- Incremental check latency after a one-line edit is reported and gated on the examples corpus.
- Incremental and full checks produce identical diagnostics on every corpus program.
- Integration test binary `issue_ai_03_agent_tooling`.

## Out of scope

No hosted service, no model calls inside the tools, no editor-specific plugins beyond a minimal VS Code client config.

## System integration impact (from S1)

Added 2026-09-26 by the integration plan (`docs/system-integration-v1.md`); existing content above is unchanged except where this section says so.

- The LSP and MCP servers cover the system dialect and graph diagnostics (S1-01, S1-03), including navigation across `connect` edges, frame-tree queries and IR-located diagnostics.
