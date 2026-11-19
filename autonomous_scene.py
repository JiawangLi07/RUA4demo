"""Procedural RGB-D sensor fixture with no recording or expert trajectory input."""
import numpy as np
from PIL import Image, ImageDraw

from hardware import SIDES, aligned_state, arm_settings, hand_settings
from rgbd import CameraGeometry, Snapshot


def object_boxes(layout):
    rng = np.random.default_rng(layout)
    x, y = int(rng.integers(65, 225)), int(rng.integers(105, 325))
    u, v = int(rng.integers(385, 515)), int(rng.integers(85, 290))
    return {"shuttlecock":[x,y,x+56,y+46], "tube":[u,v,u+52,v+100]}


def observation(layout=0):
    boxes = object_boxes(layout)
    rgb = Image.new("RGB", (640,480), (90,100,107))
    depth = Image.new("I", rgb.size, 980)
    x,y,r,b = boxes["shuttlecock"]
    u,v,s,t = boxes["tube"]
    for canvas, feather, cork, tube in ((rgb,(245,245,240),(211,178,120),(30,76,166)),
                                         (depth,792,792,864)):
        draw = ImageDraw.Draw(canvas)
        draw.polygon([(x,y+4),(x+8,y),(r,b-10),(r-12,b)], fill=feather)
        draw.ellipse([r-15,b-14,r,b],fill=cork)
        draw.rounded_rectangle([u,v,s,t],radius=9,fill=tube)
    intrinsics = dict(width=640,height=480,fx=570.,fy=570.,ppx=320.,ppy=240.,
                      coeffs=[0.]*5,model="none")
    geometry = CameraGeometry.from_metadata(dict(color=intrinsics,depth=intrinsics,
        depth_to_color_rotation=np.eye(3).ravel().tolist(),depth_to_color_translation_m=[0.]*3,
        depth_scale_m=.001))
    snap = Snapshot(np.array(rgb),np.array(depth,dtype=np.uint16),geometry,"synthetic",
                    0,0,"procedural_scene")
    arm = [(j.min_deg+j.max_deg)/2 for j in arm_settings().joints]
    hand = [(j.min_rad+j.max_rad)/2 for j in hand_settings().joints]
    return snap, aligned_state({s:arm.copy() for s in SIDES},{s:hand.copy() for s in SIDES},
                               {"virtual":0},0)


def surface_point(snapshot, label, layout):
    """Evaluator-only surface centroid for a rendered object."""
    x,y,r,b = object_boxes(layout)[label]
    depths = snapshot.raw_depth[y:b+1,x:r+1]
    rows, cols = np.nonzero(depths < 980)
    z = depths[rows,cols].astype(float)*snapshot.geometry.depth_scale_m
    camera = snapshot.geometry.color
    points = np.column_stack(((cols+x-camera.ppx)*z/camera.fx,
                              (rows+y-camera.ppy)*z/camera.fy,z))
    return np.median(points, axis=0)
