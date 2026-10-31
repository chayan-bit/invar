# Compatibility manifest inputs

Runnable vendored models now live under
`../reach_place/assets/menagerie/`. Use that complete tree for model loading.

Only the original UR5e XML and license are retained here because existing
foreign-component manifests bind their exact paths and bytes into recorded
evidence. They are identity inputs, not a complete runnable model directory.
Do not remove or rewrite them without requalifying those component contracts.

The duplicated meshes were removed only after byte comparison and MuJoCo model
equivalence checks. The upstream manifest and licenses remain in the canonical
model tree. The original full tree is recoverable at commit
`164dc101db77bbe6226de1a86f68cabfb425bc58`.

The retained XML is also the unstripped original that the vendored model derives from.
The runnable copy under `../reach_place/assets/menagerie/` drops the visual-only geoms and their meshes.
`ur5e_vendored_model_is_the_pinned_original_without_visual_geoms` proves that is the only difference.
