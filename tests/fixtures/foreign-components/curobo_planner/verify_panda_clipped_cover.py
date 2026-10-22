#!/usr/bin/env python3
"""Replay exact certificates, then compare offline covers in compiled body frames."""
import argparse
import itertools
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import scipy
from panda_clipped_cover import load_verified, sha
from test_panda_clipped_cover import collision_controls
from verify_panda_hull_cover import first_overlap


def run(args):
    if (mujoco.__version__, np.__version__, scipy.__version__) != ('3.12.0','2.5.3','1.17.0'):
        raise ValueError('pinned CPU packages required')
    start = time.monotonic()
    manifest, geometries = load_verified(args.candidate, args.panda_directory)
    certificate_seconds = time.monotonic()-start
    golden = next(a for a in json.loads(args.goldens.read_bytes())['arms'] if a['arm']=='panda')
    if golden['model_sha256'] != manifest['source_sha256']['panda.xml']:
        raise ValueError('golden identity')
    model = mujoco.MjModel.from_xml_path(str(args.panda_directory/'panda.xml'))
    expected = set(np.flatnonzero(model.geom_group==3))
    if len(expected) != 23:
        raise ValueError('compiled collision count')
    rows = []; source_penetrations = 0
    for pose, vector in enumerate(golden['vectors']):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'))
        for name,q in zip(golden['joint_names'],vector['q_rad'],strict=True):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            data.qpos[model.jnt_qposadr[jid]]=float(q)
        mujoco.mj_forward(model,data)
        by_body={}; seen=set(); worst=-float('inf'); count=0
        for geom, spheres in geometries:
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,geom['body'])
            gid=int(model.body_geomadr[body])+geom['source_geom_index'];seen.add(gid)
            if gid not in expected or model.geom_bodyid[gid]!=body:
                raise ValueError('compiled geometry mapping')
            if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(model.geom_dataid[gid]);offset=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh])
                points=model.mesh_vert[offset:offset+n]
            elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                points=np.array(list(itertools.product(*[[-v,v] for v in model.geom_size[gid]])))
            else:
                raise ValueError('unsupported compiled primitive')
            world=points@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            transformed=spheres.copy()
            transformed[:,:3]=spheres[:,:3]@data.xmat[body].reshape(3,3).T+data.xpos[body]
            by_body.setdefault(body,[]).append(transformed)
            distances=np.linalg.norm(world[:,None,:]-transformed[None,:,:3],axis=2)-transformed[:,3]
            worst=max(worst,float(np.min(distances,axis=1).max()));count+=len(points)
        if seen!=expected or worst>1e-9:
            raise ValueError('compiled primitive coverage failure')
        by_body={b:np.concatenate(parts) for b,parts in by_body.items()}
        overlaps=[]; pairs=0
        for a,b in itertools.combinations(sorted(by_body),2):
            wa,wb=int(model.body_weldid[a]),int(model.body_weldid[b])
            if (wa==wb or int(model.body_weldid[model.body_parentid[wa]])==wb
                    or int(model.body_weldid[model.body_parentid[wb]])==wa):
                continue
            pairs+=1; gap=first_overlap(by_body[a],by_body[b])
            ga=[g for g in seen if model.geom_bodyid[g]==a]
            gb=[g for g in seen if model.geom_bodyid[g]==b]
            distance=min(float(mujoco.mj_geomDistance(model,data,g,h,1.,None)) for g in ga for h in gb)
            source_penetrations+=distance<0
            if gap is None and distance<0:
                raise ValueError('source collision missed by candidate')
            if gap is not None:
                overlaps.append({'bodies':[model.body(a).name,model.body(b).name],
                                 'source_geom_distance_m':distance,
                                 'first_sphere_overlap_witness_m':gap,'false_positive':distance>1e-9})
        rows.append({'pose_index':pose,'compiled_points_checked':count,'worst_outside_m':worst,
                     'nonadjacent_body_pairs_checked':pairs,'overlaps':overlaps})
    controls=collision_controls()
    if [r['sphere_overlap'] for r in controls]!=[True,True,False] or not (controls[0]['source_distance_m']<0<controls[1]['source_distance_m']):
        raise ValueError('synthetic penetration/false-positive controls failed')
    result={'status':'offline_coverage_checks_pass_feasibility_not_accepted',
            'qualification':'candidate only; no planner-frame/cuRobo/live ROS qualification',
            'candidate_sha256':sha(args.candidate.read_bytes()),'goldens_sha256':sha(args.goldens.read_bytes()),
            'verifier_sha256':sha(Path(__file__).read_bytes()),
            'overlap_checker_sha256':sha(Path(__file__).with_name('verify_panda_hull_cover.py').read_bytes()),
            'control_source_sha256':sha(Path(__file__).with_name('test_panda_clipped_cover.py').read_bytes()),
            'spheres':sum(len(s) for _,s in geometries), 'partition_depth':manifest['partition_depth'],
            'max_sphere_radius_m':max(float(s[:,3].max()) for _,s in geometries),
            'source_penetrations_in_robot_cases':int(source_penetrations),
            'robot_cases_establish_empirical_penetration_recall':source_penetrations>0,
            'synthetic_controls':controls,'certificate_replay_seconds':certificate_seconds,
            'wall_seconds':time.monotonic()-start,'numpy':np.__version__,'scipy':scipy.__version__,
            'mujoco':mujoco.__version__,'cases':rows}
    with args.output.open('x') as handle:
        handle.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','synthetic_controls']}))
    print('false-positive counts',[sum(o['false_positive'] for o in r['overlaps']) for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['candidate','panda-directory','goldens','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
