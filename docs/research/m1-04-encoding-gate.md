# Historical M1-04 primitive encoding gate

On 2026-09-23, M1-04 was revised to require demand-aware encoding into the
typed graph, not a translation into Lafont's three primitive agents. The
untyped reducer in `src/interaction.rs` and the typed graph in
`src/typed_net.rs` are independently executable. There is no checked
source-to-three-agent encoding or decoding relation between them.

The earlier gate would have required finite gamma/delta/epsilon templates for
each admitted source constructor, plus a demanded-step relation preserving
call-by-value order and affine use. Lafont's
[*Interaction Combinators*](https://doi.org/10.1006/inco.1997.2643)
establishes universality of the primitive system, while Sinot's
[*Call-by-Name and Call-by-Value as Token-Passing Interaction Nets*](https://lsv.ens-paris-saclay.fr/Publis/PAPERS/PDF/sinot-tlca05.pdf)
uses additional agents to control evaluation order. Neither is a tested
Invar-specific primitive encoding. This research question remains open and
must not be inferred from M1 acceptance.

The accepted M1 typed rules and evidence are in
[`docs/typed-interaction-v1.md`](../typed-interaction-v1.md). M1-04 still
requires source/typed-graph demand agreement and M1-05 still requires real
typed graph rewrites and extraction.
