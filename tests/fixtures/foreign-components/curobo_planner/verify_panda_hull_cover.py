#!/usr/bin/env python3
"""CPU compiled-frame and pairwise collision diagnostics for certified sphere files."""
import argparse
import itertools
import json
from pathlib import Path
import time
import mujoco
import numpy as np
import scipy
from scipy.spatial import cKDTree
from panda_hull_cover import load_verified,sha


def first_overlap(a,b):
    tree=cKDTree(b[:,:3]);maximum=float(b[:,3].max())
    for start in range(0,len(a),128):
        block=a[start:start+128]
        neighborhoods=tree.query_ball_point(block[:,:3],block[:,3]+maximum,workers=1)
        for sphere,indices in zip(block,neighborhoods,strict=True):
            if not indices:continue
            indices=np.array(sorted(indices));other=b[indices]
            gaps=np.linalg.norm(other[:,:3]-sphere[:3],axis=1)-other[:,3]-sphere[3]
            hit=np.flatnonzero(gaps<0)
            if len(hit):return float(gaps[hit[0]])
    return None


def run(args):
    if (mujoco.__version__,np.__version__,scipy.__version__)!=('3.12.0','2.5.3','1.17.0'):
        raise ValueError('pinned CPU packages required')
    start=time.monotonic();manifest,geometries=load_verified(args.candidate,args.panda_directory)
    goldens=json.loads(args.goldens.read_bytes());golden=next(a for a in goldens['arms'] if a['arm']=='panda')
    if golden['model_sha256']!=manifest['source_sha256']['panda.xml']:raise ValueError('golden identity')
    model=mujoco.MjModel.from_xml_path(str(args.panda_directory/'panda.xml'))
    if sum(model.geom_group==3)!=23:raise ValueError('compiled collision count')
    rows=[]
    for pose,vector in enumerate(golden['vectors']):
        data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'))
        for name,q in zip(golden['joint_names'],vector['q_rad'],strict=True):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name);data.qpos[model.jnt_qposadr[jid]]=float(q)
        mujoco.mj_forward(model,data); by_body={};worst=-float('inf');points_checked=0;seen=set()
        for geom,spheres in geometries:
            body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,geom['body'])
            gid=int(model.body_geomadr[body])+geom['source_geom_index'];seen.add(gid)
            if model.geom_group[gid]!=3 or model.geom_bodyid[gid]!=body:raise ValueError('compiled geom mapping')
            if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(model.geom_dataid[gid]);offset=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
                points=model.mesh_vert[offset:offset+count]
            elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                points=np.array(list(itertools.product(*[[-v,v] for v in model.geom_size[gid]])))
            else:raise ValueError('unsupported compiled primitive')
            world=points@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            transformed=spheres.copy();transformed[:,:3]=spheres[:,:3]@data.xmat[body].reshape(3,3).T+data.xpos[body]
            by_body.setdefault(body,[]).append(transformed)
            for offset in range(0,len(world),64):
                distances=np.linalg.norm(world[offset:offset+64,None,:]-transformed[None,:,:3],axis=2)-transformed[:,3]
                worst=max(worst,float(np.min(distances,axis=1).max()))
            points_checked+=len(points)
        if seen!=set(np.flatnonzero(model.geom_group==3)):raise ValueError('compiled primitives omitted')
        if worst>1e-9:raise ValueError('compiled vertex/corner coverage failure')
        by_body={b:np.concatenate(parts) for b,parts in by_body.items()};overlaps=[];pairs=0
        for a,b in itertools.combinations(sorted(by_body),2):
            wa,wb=int(model.body_weldid[a]),int(model.body_weldid[b])
            if wa==wb or int(model.body_weldid[model.body_parentid[wa]])==wb or int(model.body_weldid[model.body_parentid[wb]])==wa:continue
            pairs+=1;gap=first_overlap(by_body[a],by_body[b])
            ga=[g for g in seen if model.geom_bodyid[g]==a];gb=[g for g in seen if model.geom_bodyid[g]==b]
            source_distance=min(float(mujoco.mj_geomDistance(model,data,g,h,1.,None)) for g in ga for h in gb)
            if gap is None and source_distance<0:
                raise ValueError('source collision missed by candidate')
            if gap is not None:
                overlaps.append({'bodies':[model.body(a).name,model.body(b).name],'first_sphere_overlap_witness_m':gap,
                                 'source_geom_distance_m':source_distance,'false_positive':source_distance>1e-9})
        rows.append({'golden_pose_index':pose,'compiled_points_checked':points_checked,'worst_outside_m':worst,
                     'nonadjacent_body_pairs_checked':pairs,'overlaps':overlaps})
    result={'status':'offline_checks_pass','qualification':'not_production_ready_no_curobo_execution',
            'candidate_sha256':sha(args.candidate.read_bytes()),'goldens_sha256':sha(args.goldens.read_bytes()),
            'verifier_sha256':sha(Path(__file__).read_bytes()),'boundary_cell_max_edge_m':manifest['boundary_cell_max_edge_m'],
            'spheres':sum(len(s) for _,s in geometries),'nodes':sum(g['certificate']['nodes'] for g in manifest['geometries']),
            'max_sphere_radius_m':max(float(s[:,3].max()) for _,s in geometries),
            'minimum_cell_corner_clearance_m':min(g['certificate']['minimum_cell_corner_clearance_m'] for g in manifest['geometries']),
            'max_support_plane_excess_m_not_euclidean_distance':max(g['certificate']['maximum_support_plane_excess_m_not_euclidean_distance'] for g in manifest['geometries']),
            'numpy':np.__version__,'scipy':scipy.__version__,'mujoco':mujoco.__version__,'wall_seconds':time.monotonic()-start,'cases':rows}
    with args.output.open('x') as handle:handle.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}));print('overlap counts',[len(r['overlaps']) for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['candidate','panda-directory','goldens','output']:parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
