{
  description = "Invar robotics integration dependencies";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/26787be7438dca718f12f4f1d10460214f59b72b";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, nix-ros-overlay, rust-overlay, ... }:
    let
      # L1-06 binding shell: the rust-toolchain.toml compiler, maturin, the
      # pytest dependencies and LLVM 21 for `--features native`.
      pythonShell = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          llvm = pkgs.llvmPackages_21.llvm;
        in
        pkgs.mkShell {
          packages = [
            (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            pkgs.maturin
            (pkgs.python3.withPackages (ps: [ ps.numpy ps.pytest ps.jsonschema ]))
            llvm.dev
          ];
          # Linked by llvm-sys for `--features native`.
          buildInputs = [ pkgs.libffi ] ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isDarwin [ pkgs.libiconv ];
          LLVM_SYS_211_PREFIX = "${llvm.dev}";
        };
      # Contributor shell: the rust-toolchain.toml compiler, cargo-nextest at the
      # version CI and CONTRIBUTING.md pin, just, and the platform C/C++ toolchain
      # (mkShell supplies `cc`, `c++` and the linker).
      rustShell = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          # nixpkgs lags the pinned version; keep its derivation and patches.
          nextest = pkgs.cargo-nextest.overrideAttrs (old: rec {
            version = "0.9.146";
            src = pkgs.fetchFromGitHub {
              owner = "nextest-rs";
              repo = "nextest";
              tag = "cargo-nextest-${version}";
              hash = "sha256-2ndrHZ1pmu1oBZGlq5pyv0vhSEUf23gNRyyenNw0fdk=";
            };
            cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
              inherit src;
              hash = "sha256-ZW0/JQ9RECxfDn479ww4bO4ixoMGaM+GtGUOCWy2Cyg=";
            };
          });
        in
        pkgs.mkShell {
          packages = [
            (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            nextest
            pkgs.just
          ];
        };
      rosShells = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ nix-ros-overlay.overlays.default ];
          };
          ros = pkgs.rosPackages.lyrical;
        in
        {
          ros2 = pkgs.mkShell {
            packages = [
              (ros.buildEnv {
                underlay = true;
                paths = [
                  pkgs.colcon
                  ros.ros-core
                  ros.ros2-control
                  ros.ros2-controllers
                  ros.controller-manager
                  ros.mujoco-ros2-control
                  ros.mujoco-ros2-control-demos
                  ros.mujoco-vendor
                  ros.generate-parameter-library
                  ros.rmw-zenoh-cpp
                  ros.rosbag2-storage-mcap
                  ros.ros2launch
                  ros.pinocchio
                ];
              })
              (pkgs.python3.withPackages (ps: [ ps.mujoco ]))
              pkgs.uv
            ];
          };
        };
    in
    {
      devShells =
        nixpkgs.lib.genAttrs [ "x86_64-linux" "aarch64-linux" ]
          (system: rosShells system // { python = pythonShell system; default = rustShell system; })
        // { aarch64-darwin = { python = pythonShell "aarch64-darwin"; default = rustShell "aarch64-darwin"; }; };
    };

  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
  };
}
