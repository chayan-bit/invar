#!/usr/bin/env python3
"""Additive CPU UR5e source/compiled energy witness; never replaces frozen evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin
from attribute_mass import jacobian, numbers
from run import full_mass, mj_data, pin_mass, pin_q

ROOT = Path(__file__).resolve().parent
FROZEN = {'contract.json': '85362985e3729e6f40eac401efc3772d27fb90f93238f3e0174a01517bd2b0b6',
          'reference.json': '8a4cef46ff9aeec67ab51f5c426b219e0d467eb16d0921679586ebc6cceb64c4',
          'attribution.json': '3350f84ca3fb3f0f30d322a76c06a6d978a8cf140e57afb5244f6d1ccb0c5a91'}
TOLERANCE = 1e-10  # Existing source-energy attribution criterion, not GPU tolerance.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(text):
    q = np.array([float(x) for x in text.split()])
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('invalid source wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def source_bodies(source, model):
    result = []
    for body in ET.parse(source).getroot().iter('body'):
        inertia = body.find('inertial')
        if inertia is None or set(inertia.attrib) - {'mass', 'pos', 'quat', 'diaginertia'}:
            raise ValueError('requires explicit diagonal inertia with optional quaternion')
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.attrib['name'])
        if body_id < 1:
            raise ValueError('missing compiled body')
        rot = rotation(inertia.get('quat', '1 0 0 0'))
        diagonal = np.fromstring(inertia.attrib['diaginertia'], sep=' ')
        com = np.fromstring(inertia.attrib['pos'], sep=' ')
        mass = float(inertia.attrib['mass'])
        if diagonal.shape != (3,) or com.shape != (3,) or not np.isfinite(com).all() or \
                not np.isfinite(diagonal).all() or np.any(diagonal <= 0) or not np.isfinite(mass) or mass <= 0:
            raise ValueError('invalid source inertial values')
        result.append((body_id, mass, com, rot @ np.diag(diagonal) @ rot.T))
    if len(result) != model.nbody - 1:
        raise ValueError('source/compiled body count mismatch')
    return result


def residual(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (6, 6) or b.shape != (6, 6) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid mass matrix')
    return float(np.max(np.abs(a - b)))


def run(models_root, output):
    for name, expected in FROZEN.items():
        if sha(ROOT / name) != expected:
            raise ValueError('frozen evidence changed: ' + name)
    if (pin.__version__, mujoco.__version__, np.__version__) != ('4.1.0', '3.12.0', '2.5.3'):
        raise ValueError('requires Pinocchio4.1.0 MuJoCo3.12.0 NumPy2.5.3')
    contract, reference = [json.loads((ROOT / name).read_text()) for name in ['contract.json', 'reference.json']]
    spec = next(x for x in contract['models'] if x['name'] == 'ur5e')
    prior = next(x for x in reference['models'] if x['name'] == 'ur5e')
    source = models_root / spec['file']
    if sha(source) != spec['sha256'] or sha(source.parent / 'LICENSE') != spec['license_sha256']:
        raise ValueError('pinned source/license mismatch')
    if [c['name'] for c in prior['cases']] != contract['inputs']['ordered_cases']:
        raise ValueError('case order mismatch')
    model = mujoco.MjModel.from_xml_path(str(source))
    pmodel = pin.buildModelFromMJCF(str(source))
    pdata = pmodel.createData()
    bodies = source_bodies(source, model)
    dofs = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in spec['active_joints']]
    armature = np.diag(np.array(spec['source_armature_diag_text'], dtype=float))
    cases = []
    for case in prior['cases']:
        q = list(map(float, case['q_rad']))
        data = mj_data(model, spec, q)
        source_m, compiled_m = armature.copy(), armature.copy()
        for body_id, mass, com, inertia in bodies:
            rot, origin = data.xmat[body_id].reshape(3, 3), data.xpos[body_id]
            jv, jw = jacobian(model, data, origin + rot @ com, body_id, dofs)
            source_m += mass * jv.T @ jv + jw.T @ (rot @ inertia @ rot.T) @ jw
            cr = rotation(' '.join(map(str, model.body_iquat[body_id])))
            ci = cr @ np.diag(model.body_inertia[body_id]) @ cr.T
            jv, jw = jacobian(model, data, origin + rot @ model.body_ipos[body_id], body_id, dofs)
            compiled_m += model.body_mass[body_id] * jv.T @ jv + jw.T @ (rot @ ci @ rot.T) @ jw
        pbody, _ = pin_mass(pmodel, pdata, spec, pin_q(pmodel, spec, q))
        fresh_pin = pbody + armature
        mraw, imported = full_mass(model, data, spec)
        fresh_mj = mraw - np.diag(imported) + armature
        errors = {name: residual(a, b) for name, a, b in [
            ('source_vs_fresh_pinocchio', source_m, fresh_pin),
            ('compiled_vs_fresh_mujoco', compiled_m, fresh_mj),
            ('fresh_pinocchio_vs_frozen', fresh_pin, case['mass_pinocchio']),
            ('fresh_mujoco_vs_frozen', fresh_mj, case['mass_mujoco'])]}
        cases.append({'name': case['name'], 'q_rad': case['q_rad'],
                      'source_kinetic_mass': numbers(source_m), 'compiled_kinetic_mass': numbers(compiled_m),
                      'fresh_pinocchio': numbers(fresh_pin), 'fresh_mujoco': numbers(fresh_mj),
                      'max_abs_residuals': errors})
    maximum = max(v for c in cases for v in c['max_abs_residuals'].values())
    record = {'schema': 'invar-ur5e-source-energy-witness-v1', 'status': 'pass' if maximum < TOLERANCE else 'failed',
              'scope': 'fresh cloud x86_64 CPU reconstruction; not ROS or GPU qualification',
              'method': 'sum source mass Jv_com.T Jv_com + Jw.T R I_com R.T Jw; normalized source wxyz rotation; source armature once',
              'criterion': {'max_abs_strictly_less_than': TOLERANCE, 'origin': 'existing attribute_mass.py criterion'},
              'frozen_sha256': FROZEN, 'source_sha256': sha(source), 'script_sha256': sha(Path(__file__)),
              'helper_sha256': {n: sha(ROOT / n) for n in ['run.py', 'attribute_mass.py']},
              'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'active_joints': spec['active_joints'], 'source_armature_diag_text': spec['source_armature_diag_text'],
              'max_abs_residual': maximum, 'cases': cases}
    with output.open('x') as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'cases': len(cases), 'max_abs_residual': maximum}))
    return 0 if record['status'] == 'pass' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.models_root, args.output))
