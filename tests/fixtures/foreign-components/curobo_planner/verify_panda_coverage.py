#!/usr/bin/env python3
"""Independent compiled-MuJoCo frame/vertex check of offline coverage proposals."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from panda_coverage_candidate import MODEL_SHA, certify, source_geometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--panda-directory',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--goldens',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if mujoco.__version__!='3.12.0' or np.__version__!='2.5.3':
        raise ValueError('requires pinned MuJoCo3.12.0 NumPy2.5.3')
    report=json.loads(args.candidate.read_bytes())
    for name,digest in report['source_sha256'].items():
        if sha(args.panda_directory/name)!=digest:
            raise ValueError('source digest mismatch')
    if sha(args.panda_directory/'panda.xml')!=MODEL_SHA:
        raise ValueError('source model mismatch')
    golden=next(a for a in json.loads(args.goldens.read_bytes())['arms'] if a['arm']=='panda')
    if golden['model_sha256']!=MODEL_SHA:
        raise ValueError('golden model mismatch')
    model=mujoco.MjModel.from_xml_path(str(args.panda_directory/'panda.xml'))
    if sum(model.geom_group==3)!=23:
        raise ValueError('compiled collision geometry count mismatch')
    source_geoms,_=source_geometry(args.panda_directory)
    expected={(g['body'],g['source_geom_index']):(g['lo'],g['hi']) for g in source_geoms}
    records=[]
    for variant in ['single_sphere','axial_cells']:
        geometries=report[variant]['geometries']
        identities=[(g['body'],g['source_geom_index']) for g in geometries]
        if len(identities)!=len(set(identities)) or set(identities)!=set(expected):
            raise ValueError('missing/duplicate source collision geometry')
        if any((g['lo'],g['hi'])!=expected[(g['body'],g['source_geom_index'])] for g in geometries):
            raise ValueError('candidate box differs from source bounds')
        for pose_index,vector in enumerate(golden['vectors']):
            data=mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model,data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'))
            for name,value in zip(golden['joint_names'],vector['q_rad'],strict=True):
                joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                data.qpos[model.jnt_qposadr[joint]]=float(value)
            mujoco.mj_forward(model,data)
            checked=0; worst=-float('inf'); spheres_by_body={}
            for geom in report[variant]['geometries']:
                certify(geom['lo'],geom['hi'],geom['axis'],geom['cells'])
                body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,geom['body'])
                gid=int(model.body_geomadr[body])+geom['source_geom_index']
                if model.geom_bodyid[gid]!=body or model.geom_group[gid]!=3:
                    raise ValueError('compiled source-geometry mapping mismatch')
                if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=int(model.geom_dataid[gid]); start=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
                    points=model.mesh_vert[start:start+count]
                elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                    points=np.array(list(itertools.product(*[[-v,v] for v in model.geom_size[gid]])))
                else:
                    raise ValueError('unexpected compiled collision shape')
                world=np.asarray(points)@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
                centers=np.array([c['center'] for c in geom['cells']])@data.xmat[body].reshape(3,3).T+data.xpos[body]
                radii=np.array([c['radius'] for c in geom['cells']])
                outside=np.linalg.norm(world[:,None,:]-centers[None,:,:],axis=2)-radii
                worst=max(worst,float(np.min(outside,axis=1).max())); checked+=len(points)
                spheres_by_body.setdefault(body,[]).extend(zip(centers,radii))
            if worst>1e-9:
                raise ValueError('compiled mesh vertex/box corner outside candidate')
            # Diagnostic only: no collision ignore policy is changed here.
            # Counts exclude same/parent welded bodies, mirroring structural adjacency.
            overlaps=[]
            for a,b in itertools.combinations(sorted(spheres_by_body),2):
                wa,wb=int(model.body_weldid[a]),int(model.body_weldid[b])
                if wa==wb or int(model.body_weldid[model.body_parentid[wa]])==wb or int(model.body_weldid[model.body_parentid[wb]])==wa:
                    continue
                separation=min(float(np.linalg.norm(ca-cb))-ra-rb for ca,ra in spheres_by_body[a] for cb,rb in spheres_by_body[b])
                if separation<0:
                    ga=[g for g in range(model.ngeom) if model.geom_bodyid[g]==a and model.geom_group[g]==3]
                    gb=[g for g in range(model.ngeom) if model.geom_bodyid[g]==b and model.geom_group[g]==3]
                    distance=min(float(mujoco.mj_geomDistance(model,data,g,h,1.0,None)) for g in ga for h in gb)
                    overlaps.append({'bodies':[model.body(a).name,model.body(b).name],
                                     'minimum_sphere_separation_m':separation,
                                     'minimum_compiled_source_geom_distance_m':distance,
                                     'source_pair_separated_beyond_audit_threshold':distance>1e-9})
            records.append({'variant':variant,'golden_pose_index':pose_index,'points_checked':checked,
                            'worst_outside_m':worst,'nonadjacent_sphere_overlap_pairs_diagnostic_only':overlaps})
    result={'status':'pass','scope':'offline source coverage/frame preparation only; no collision-policy or planner acceptance',
            'candidate_sha256':sha(args.candidate),'goldens_sha256':sha(args.goldens),'mujoco':mujoco.__version__,
            'numpy':np.__version__,'python':sys.version,'cases':records}
    with args.output.open('x') as handle: handle.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'pass','pose_variant_checks':len(records),'points_checked':sum(r['points_checked'] for r in records),
                      'worst_outside_m':max(r['worst_outside_m'] for r in records),
                      'overlap_pair_counts':[len(r['nonadjacent_sphere_overlap_pairs_diagnostic_only']) for r in records]}))


if __name__=='__main__':main()
