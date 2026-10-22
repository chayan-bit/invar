//! Robotics suite: batched forward kinematics, a damped least-squares IK
//! step, sphere-set collision checks, MPC rollouts and point-cloud voxel and
//! radius filters, as ordinary loops.
use super::{Case, Input, Kernel, Wanted, oracle};

fn input(len: u32, seed: u32) -> Input {
    Input { len, seed }
}

fn batched_fk() -> Kernel {
    let cases = [(1, 1, 111), (8, 3, 112), (33, 6, 113), (64, 7, 114)]
        .map(|(n, links, seed)| {
            let q = oracle::fill(n * links, seed);
            Case {
                shape: vec![("n", n), ("l", links)],
                inputs: vec![input(n * links, seed)],
                call: format!("batched_fk(in0, {n}u32, {links}u32)"),
                wanted: Wanted::F32(oracle::batched_fk(&q, n as usize, links as usize)),
            }
        })
        .into();
    Kernel {
        name: "batched_fk",
        leaf: include_str!("batched_fk.inv"),
        cases,
    }
}

fn dls_ik_step() -> Kernel {
    let cases = [(1, 121), (7, 122), (32, 123), (65, 124)]
        .map(|(n, seed)| {
            let q = oracle::fill(n * 3, seed);
            let target = oracle::fill(n * 2, seed + 100);
            Case {
                shape: vec![("n", n)],
                inputs: vec![input(n * 3, seed), input(n * 2, seed + 100)],
                call: format!("dls_ik_step(in0, in1, {n}u32)"),
                wanted: Wanted::F32(oracle::dls_ik_step(&q, &target, n as usize)),
            }
        })
        .into();
    Kernel {
        name: "dls_ik_step",
        leaf: include_str!("dls_ik_step.inv"),
        cases,
    }
}

fn sphere_collision() -> Kernel {
    let cases = [(1, 1, 131), (9, 14, 132), (40, 25, 133), (48, 40, 134)]
        .map(|(na, nb, seed)| {
            let a = oracle::fill(na * 4, seed);
            let b = oracle::fill(nb * 4, seed + 100);
            Case {
                shape: vec![("a", na), ("b", nb)],
                inputs: vec![input(na * 4, seed), input(nb * 4, seed + 100)],
                call: format!("sphere_collision(in0, in1, {na}u32, {nb}u32)"),
                wanted: Wanted::U32(oracle::sphere_collision(&a, &b, na as usize, nb as usize)),
            }
        })
        .into();
    Kernel {
        name: "sphere_collision",
        leaf: include_str!("sphere_collision.inv"),
        cases,
    }
}

fn mpc_rollout() -> Kernel {
    let cases = [(1, 1, 141), (16, 10, 142), (33, 12, 143), (40, 12, 144)]
        .map(|(k, horizon, seed)| {
            let controls = oracle::fill(k * horizon * 2, seed);
            Case {
                shape: vec![("k", k), ("h", horizon)],
                inputs: vec![input(k * horizon * 2, seed)],
                call: format!("mpc_rollout(in0, {k}u32, {horizon}u32)"),
                wanted: Wanted::F32(oracle::mpc_rollout(&controls, k as usize, horizon as usize)),
            }
        })
        .into();
    Kernel {
        name: "mpc_rollout",
        leaf: include_str!("mpc_rollout.inv"),
        cases,
    }
}

fn voxel_filter() -> Kernel {
    let cases = [(1, 1, 151), (50, 2, 152), (200, 4, 153), (300, 3, 154)]
        .map(|(n, g, seed)| {
            let points = oracle::fill(n * 3, seed);
            Case {
                shape: vec![("n", n), ("g", g)],
                inputs: vec![input(n * 3, seed)],
                call: format!("voxel_filter(in0, {n}u32, {g}u32)"),
                wanted: Wanted::F32(oracle::voxel_filter(&points, n as usize, g)),
            }
        })
        .into();
    Kernel {
        name: "voxel_filter",
        leaf: include_str!("voxel_filter.inv"),
        cases,
    }
}

fn radius_filter() -> Kernel {
    let cases = [
        (1, "0.5", 161),
        (20, "0.1", 162),
        (48, "0.05", 163),
        (64, "0.3", 164),
    ]
    .map(|(n, r2, seed)| {
        let points = oracle::fill(n * 3, seed);
        Case {
            shape: vec![("n", n)],
            inputs: vec![input(n * 3, seed)],
            call: format!("radius_filter(in0, {n}u32, {r2})"),
            wanted: Wanted::U32(oracle::radius_filter(
                &points,
                n as usize,
                r2.parse::<f32>().unwrap(),
            )),
        }
    })
    .into();
    Kernel {
        name: "radius_filter",
        leaf: include_str!("radius_filter.inv"),
        cases,
    }
}

pub fn kernels() -> Vec<Kernel> {
    vec![
        batched_fk(),
        dls_ik_step(),
        sphere_collision(),
        mpc_rollout(),
        voxel_filter(),
        radius_filter(),
    ]
}
