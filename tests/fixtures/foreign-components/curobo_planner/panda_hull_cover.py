#!/usr/bin/env python3
"""Bounded offline sphere-cover proposal with a replayable box-tree certificate."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time
import zipfile

import numpy as np
import scipy
from scipy.spatial import ConvexHull
from panda_coverage_candidate import MARGIN, source_geometry

MAX_NODES = 1000000
MAX_SPHERES = 200000
MAX_SECONDS = 90


def sha(data):
    return hashlib.sha256(data).hexdigest()


def planes(points):
    points=np.unique(np.asarray(points,dtype=np.float64),axis=0)
    hull=ConvexHull(points)
    normals=hull.equations[:,:3]
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    # Source support maxima, not rounded hull offsets: each half-space contains
    # every original vertex, hence their whole convex hull.
    supports=np.max(points@normals.T,axis=0)
    return normals,supports


def children(lo,hi):
    axis=int(np.argmax(hi-lo)); middle=(lo[axis]+hi[axis])/2
    lower_hi=hi.copy();lower_hi[axis]=middle
    upper_lo=lo.copy();upper_lo[axis]=middle
    yield lo,lower_hi
    yield upper_lo,hi


def metrics(lo,hi,normals,supports):
    center=(lo+hi)/2; half=(hi-lo)/2
    radius=float(np.linalg.norm(half))+MARGIN
    # L1 support function of the AABB yields the exact plane minimum, up to
    # bounded binary64 evaluation. The existing 1um margin dominates roundoff.
    minimum=normals@center-np.abs(normals)@half-supports
    ball_outside=normals@center+radius-supports
    return center,radius,float(minimum.max()),float(ball_outside.max())


def generate(geom,boundary_step,deadline):
    normals,supports=planes(geom['points'])
    tree=bytearray(); spheres=[]; stats={'outside':0,'interior':0,'boundary':0,'nodes':0,'max_depth':0}
    def visit(lo,hi,depth):
        stats['nodes']+=1;stats['max_depth']=max(stats['max_depth'],depth)
        if stats['nodes']>MAX_NODES or len(spheres)>MAX_SPHERES or depth>40 or time.monotonic()>deadline:
            raise ValueError('offline cover preparation bound exceeded')
        center,radius,outside,ball_outside=metrics(lo,hi,normals,supports)
        if outside>MARGIN:
            tree.append(ord('X'));stats['outside']+=1
        elif ball_outside < -MARGIN or float(np.max(hi-lo))<=boundary_step:
            tree.append(ord('S'));spheres.append([*center,radius])
            stats['interior' if ball_outside < -MARGIN else 'boundary']+=1
        else:
            tree.append(ord('I'))
            for a,b in children(lo,hi):visit(a,b,depth+1)
    visit(np.array(geom['lo']),np.array(geom['hi']),0)
    data=np.array(spheres,dtype='<f8').reshape(-1,4).tobytes()
    return bytes(tree),data,stats


def certify(geom,boundary_step,tree,raw):
    normals,supports=planes(geom['points'])
    if len(tree)>MAX_NODES or len(raw)%32 or len(raw)//32>MAX_SPHERES:
        raise ValueError('certificate size bound')
    spheres=np.frombuffer(raw,dtype='<f8').reshape(-1,4)
    if not np.isfinite(spheres).all():raise ValueError('nonfinite sphere')
    node=sphere=0;minimum_clearance=math.inf; max_plane_excess=0.
    def visit(lo,hi,depth):
        nonlocal node,sphere,minimum_clearance,max_plane_excess
        if node>=len(tree) or depth>40:raise ValueError('incomplete box-tree')
        code=tree[node];node+=1
        center,radius,outside,ball_outside=metrics(lo,hi,normals,supports)
        if code==ord('X'):
            if outside<=MARGIN:raise ValueError('discarded cell lacks separation proof')
        elif code==ord('S'):
            if sphere>=len(spheres):raise ValueError('missing sphere')
            actual=spheres[sphere];sphere+=1
            # Verify actual center/radius against every corner via the farthest
            # corner formula, without trusting the generator radius.
            distance=float(np.linalg.norm(np.maximum(np.abs(lo-actual[:3]),np.abs(hi-actual[:3]))))
            if distance>actual[3]:raise ValueError('sphere does not enclose its solid cell')
            excess=float((normals@actual[:3]+actual[3]-supports).max())
            if excess>=-MARGIN and float(np.max(hi-lo))>boundary_step:
                raise ValueError('boundary refinement criterion violated')
            minimum_clearance=min(minimum_clearance,float(actual[3])-distance)
            max_plane_excess=max(max_plane_excess,excess)
        elif code==ord('I'):
            for a,b in children(lo,hi):visit(a,b,depth+1)
        else:raise ValueError('unknown box-tree node')
    visit(np.array(geom['lo']),np.array(geom['hi']),0)
    if node!=len(tree) or sphere!=len(spheres):raise ValueError('unused certificate data')
    return {'spheres':sphere,'nodes':node,'minimum_cell_corner_clearance_m':minimum_clearance,
            'maximum_support_plane_excess_m_not_euclidean_distance':max_plane_excess}


def build(directory,step):
    if step not in (0.01,0.005):raise ValueError('only bounded 10mm/5mm candidate preparation')
    if (np.__version__,scipy.__version__)!=('2.5.3','1.17.0'):raise ValueError('pinned numerical packages required')
    geometry,hashes=source_geometry(directory); files={}; records=[];deadline=time.monotonic()+MAX_SECONDS
    for i,geom in enumerate(geometry):
        tree,raw,stats=generate(geom,step,deadline)
        certificate=certify(geom,step,tree,raw)
        files[f'{i}.tree']=tree;files[f'{i}.f64le']=raw
        records.append({k:v for k,v in geom.items() if k!='points'}|{'index':i,'stats':stats,'certificate':certificate})
    if sum(r['certificate']['spheres'] for r in records)>MAX_SPHERES:raise ValueError('aggregate sphere limit')
    manifest={'schema':'panda-source-hull-binary-tree-proposal-v1','status':'offline_candidate_only',
              'source_sha256':hashes,'boundary_cell_max_edge_m':step,'geometry_margin_m':MARGIN,
              'source_frame':'original Menagerie body local; planner mapping unestablished',
              'numpy':np.__version__,'scipy':scipy.__version__,'generator_sha256':sha(Path(__file__).read_bytes()),
              'qualification':'no production geometry, collision policy or cuRobo acceptance',
              'geometries':records,'files':{n:{'sha256':sha(b),'bytes':len(b)} for n,b in files.items()}}
    files['manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,9,30,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,data)
    return blob.getvalue(),manifest


def load_verified(path,directory):
    geometry,hashes=source_geometry(directory)
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())!=47 or sum(i.file_size for i in archive.infolist())>32*1024*1024:
            raise ValueError('archive bounds')
        manifest=json.loads(archive.read('manifest.json'))
        if manifest['schema']!='panda-source-hull-binary-tree-proposal-v1' or manifest['boundary_cell_max_edge_m'] not in (0.01,0.005) or manifest['geometry_margin_m']!=MARGIN:
            raise ValueError('candidate contract mismatch')
        expected={f'{i}.{suffix}' for i in range(23) for suffix in ['tree','f64le']}
        if set(archive.namelist())!=expected|{'manifest.json'} or len(set(archive.namelist()))!=47:
            raise ValueError('archive members')
        if manifest['source_sha256']!=hashes or len(manifest['geometries'])!=23:
            raise ValueError('source identity mismatch')
        if set(manifest['files'])!=expected:raise ValueError('manifest member mismatch')
        result=[]
        for i,(geom,record) in enumerate(zip(geometry,manifest['geometries'],strict=True)):
            if any(record[k]!=geom[k] for k in ['body','source_geom_index','lo','hi']):raise ValueError('geometry identity')
            tree,raw=[archive.read(f'{i}.{suffix}') for suffix in ['tree','f64le']]
            for suffix,data in [('tree',tree),('f64le',raw)]:
                if manifest['files'][f'{i}.{suffix}']!={'sha256':sha(data),'bytes':len(data)}:raise ValueError('member digest')
            certificate=certify(geom,manifest['boundary_cell_max_edge_m'],tree,raw)
            if certificate!=record['certificate']:raise ValueError('certificate summary mismatch')
            result.append((geom,np.frombuffer(raw,dtype='<f8').reshape(-1,4).copy()))
    if sum(len(spheres) for _,spheres in result)>MAX_SPHERES:raise ValueError('aggregate sphere bound')
    return manifest,result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--panda-directory',type=Path,required=True)
    parser.add_argument('--boundary-step',type=float,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();blob,manifest=build(args.panda_directory,args.boundary_step)
    with args.output.open('xb') as handle:handle.write(blob)
    print(json.dumps({'bytes':len(blob),'sha256':sha(blob),'seconds':time.monotonic()-start,
                      'spheres':sum(g['certificate']['spheres'] for g in manifest['geometries']),
                      'nodes':sum(g['certificate']['nodes'] for g in manifest['geometries'])}))
