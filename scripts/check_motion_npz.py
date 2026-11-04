"""Check a csv_to_npz.py output against its source motion: frames, fps, joint/root agreement, foot and pelvis
heights; optionally plot a few joints.

  python scripts/check_motion_npz.py <source.csv|source.npz> <converted.npz> [--source-fps 30] [--plot out.png]

Source: a CSV as read by csv_to_npz.py (root x,y,z, root quat x,y,z,w, 29 joint positions) at --source-fps, or a
npz with root_pos, root_quat (wxyz), joint_pos (N,29) and fps. The source is interpolated linearly (quaternions
normalized) onto the converted file's time grid before comparing.
"""
import argparse

import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("converted")
ap.add_argument("--source-fps", type=float, default=30.0, help="fps of a CSV source (a npz source carries its own)")
ap.add_argument("--plot", help="write a joint plot to this png")
a = ap.parse_args()

if a.source.endswith(".csv"):
  c = np.loadtxt(a.source, delimiter=",")
  root_pos, root_quat, jp, fs = c[:, :3], c[:, [6, 3, 4, 5]], c[:, 7:], a.source_fps
else:
  s = np.load(a.source, allow_pickle=True)
  root_pos, root_quat, jp, fs = s["root_pos"], s["root_quat"], s["joint_pos"], float(s["fps"])
o = np.load(a.converted)
fo = float(np.asarray(o["fps"]).ravel()[0])
n_s, n_o = len(jp), len(o["joint_pos"])
print(f"source: {n_s} frames @ {fs} fps ({(n_s - 1) / fs:.3f} s);  converted: {n_o} frames @ {fo} fps (dt {1 / fo:.4f} s)")
print({k: o[k].shape for k in o.files})

ts, to = np.arange(n_s) / fs, np.arange(n_o) / fo
to = to[to <= ts[-1] + 1e-9]
interp = lambda x: np.stack([np.interp(to, ts, x[:, i]) for i in range(x.shape[1])], 1)  # noqa: E731
rq = np.array(root_quat, dtype=np.float64)
for i in range(1, len(rq)):  # same hemisphere as the previous frame before interpolating
  if (rq[i] * rq[i - 1]).sum() < 0:
    rq[i] = -rq[i]
q = interp(rq)
q /= np.linalg.norm(q, axis=1, keepdims=True)
n = len(to)
print(f"max |joint_pos diff| over {n} frames: {np.abs(interp(jp) - o['joint_pos'][:n]).max():.2e} rad")
print(f"max |root_pos - body_pos_w[:,0]|: {np.abs(interp(root_pos) - o['body_pos_w'][:n, 0]).max():.2e} m")
print(f"max (1-|<root_quat, body_quat_w[:,0]>|): {np.abs(np.abs((q * o['body_quat_w'][:n, 0]).sum(1)) - 1).max():.2e}")

m = get_spec().compile()  # body 0 = world in the model; the npz excludes it
names = [m.body(i).name for i in range(1, m.nbody)]
for b in ("left_ankle_roll_link", "right_ankle_roll_link"):
  z = o["body_pos_w"][:, names.index(b), 2]
  print(f"{b} z: min {z.min():.3f} max {z.max():.3f} m")
print(f"pelvis z: {o['body_pos_w'][:, 0, 2].min():.3f}..{o['body_pos_w'][:, 0, 2].max():.3f} m, "
      f"xy travel {np.linalg.norm(o['body_pos_w'][-1, 0, :2] - o['body_pos_w'][0, 0, :2]):.2f} m")

if a.plot:
  import matplotlib.pyplot as plt

  joints = {0: "left_hip_pitch", 3: "left_knee", 14: "waist_pitch", 18: "left_elbow"}
  fig, ax = plt.subplots(len(joints), 1, figsize=(8, 8), sharex=True)
  for axj, (j, nm) in zip(ax, joints.items()):
    axj.plot(ts, jp[:, j], lw=3, alpha=0.4, label="source")
    axj.plot(np.arange(n_o) / fo, o["joint_pos"][:, j], lw=1, label="csv_to_npz output")
    axj.set_ylabel(nm + " [rad]")
  ax[0].legend()
  ax[-1].set_xlabel("t [s]")
  fig.tight_layout()
  fig.savefig(a.plot, dpi=80)
  print("plot:", a.plot)
