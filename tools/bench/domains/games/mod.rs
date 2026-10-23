//! Games suite: particle integration, uniform-grid broadphase, linear-blend
//! skinning, frustum culling and grid pathfinding, as ordinary loops.
use super::{Case, Input, Kernel, Wanted, oracle};

fn input(len: u32, seed: u32) -> Input {
    Input { len, seed }
}

fn particle_integrate() -> Kernel {
    let cases = [(1, 1, 61), (9, 3, 62), (33, 40, 63), (64, 10, 64)]
        .map(|(n, steps, seed)| {
            let p = oracle::fill(n * 3, seed);
            let v = oracle::fill(n * 3, seed + 100);
            Case {
                shape: vec![("n", n), ("t", steps)],
                inputs: vec![input(n * 3, seed), input(n * 3, seed + 100)],
                call: format!("particle_integrate(in0, in1, {n}u32, {steps}u32)"),
                wanted: Wanted::F32(oracle::particle_integrate(
                    &p,
                    &v,
                    n as usize,
                    steps as usize,
                )),
            }
        })
        .into();
    Kernel {
        name: "particle_integrate",
        leaf: include_str!("particle_integrate.inv"),
        cases,
    }
}

fn grid_broadphase() -> Kernel {
    let cases = [
        (1, 1, "0.5", 71),
        (24, 4, "0.09", 72),
        (60, 8, "0.04", 73),
        (80, 3, "0.25", 74),
    ]
    .map(|(n, g, r2, seed)| {
        let pos = oracle::fill(n * 2, seed);
        let wanted = oracle::grid_broadphase(&pos, n as usize, g, r2.parse::<f32>().unwrap());
        Case {
            shape: vec![("n", n), ("g", g)],
            inputs: vec![input(n * 2, seed)],
            call: format!("grid_broadphase(in0, {n}u32, {g}u32, {r2})"),
            wanted: Wanted::U32(wanted),
        }
    })
    .into();
    Kernel {
        name: "grid_broadphase",
        leaf: include_str!("grid_broadphase.inv"),
        cases,
    }
}

fn skinning() -> Kernel {
    let cases = [(1, 1, 81), (10, 3, 82), (33, 5, 83), (64, 2, 84)]
        .map(|(n, nb, seed)| {
            let verts = oracle::fill(n * 3, seed);
            let weights = oracle::fill(n * 4, seed + 100);
            let bones = oracle::fill(nb * 12, seed + 200);
            Case {
                shape: vec![("n", n), ("b", nb)],
                inputs: vec![
                    input(n * 3, seed),
                    input(n * 4, seed + 100),
                    input(nb * 12, seed + 200),
                ],
                call: format!("skinning(in0, in1, in2, {n}u32, {nb}u32)"),
                wanted: Wanted::F32(oracle::skinning(
                    &verts,
                    &weights,
                    &bones,
                    n as usize,
                    nb as usize,
                )),
            }
        })
        .into();
    Kernel {
        name: "skinning",
        leaf: include_str!("skinning.inv"),
        cases,
    }
}

fn frustum_cull() -> Kernel {
    let cases = [(1, 91), (31, 92), (100, 93), (257, 94)]
        .map(|(n, seed)| {
            let spheres = oracle::fill(n * 4, seed);
            Case {
                shape: vec![("n", n)],
                inputs: vec![input(n * 4, seed)],
                call: format!("frustum_cull(in0, {n}u32)"),
                wanted: Wanted::U32(oracle::frustum_cull(&spheres, n as usize)),
            }
        })
        .into();
    Kernel {
        name: "frustum_cull",
        leaf: include_str!("frustum_cull.inv"),
        cases,
    }
}

fn grid_bfs() -> Kernel {
    let cases = [(1, 101), (5, 102), (12, 103), (20, 104)]
        .map(|(n, seed)| {
            let walls = oracle::fill(n * n, seed);
            Case {
                shape: vec![("n", n)],
                inputs: vec![input(n * n, seed)],
                call: format!("grid_bfs(in0, {n}u32)"),
                wanted: Wanted::U32(oracle::grid_bfs(&walls, n as usize)),
            }
        })
        .into();
    Kernel {
        name: "grid_bfs",
        leaf: include_str!("grid_bfs.inv"),
        cases,
    }
}

pub fn kernels() -> Vec<Kernel> {
    vec![
        particle_integrate(),
        grid_broadphase(),
        skinning(),
        frustum_cull(),
        grid_bfs(),
    ]
}
