# Pinned Panda FK input

`franka_emika_panda/panda.xml` and its `LICENSE` are from
[`google-deepmind/mujoco_menagerie` revision `c96a32d28fb5da84da38c1da4d749e7a13212855`](https://github.com/google-deepmind/mujoco_menagerie/tree/c96a32d28fb5da84da38c1da4d749e7a13212855/franka_emika_panda).
The model is Apache-2.0 and is used only for Pinocchio's kinematic-model parser;
the visual mesh assets are not bundled here.

- `panda.xml`: SHA-256 `96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273`
- `LICENSE`: SHA-256 `a6cba85bc92e0cff7a450b1d873c0eaa2e9fc96bf472df0247a26bec77bf3ff9`

The independent Python Pinocchio 4.1.0 goldens record the same model and license
hashes. The generated S1-02 harness checks the model hash before execution.
