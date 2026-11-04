"""Convert a root-pose motion npz (root_pos, root_quat wxyz, joint_pos (N,29) in Unitree SDK order, fps)
into the LAFAN1-retargeting CSV that scripts/csv_to_npz.py reads:
one row per frame = root x,y,z, root quat x,y,z,w, 29 joint positions (same SDK joint order).
No resampling: pass the printed fps as --input-fps to csv_to_npz.py.

  python scripts/g1deploy_npz_to_csv.py <in.npz> <out.csv>
"""
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
d = np.load(src, allow_pickle=True)
pos = d["root_pos"].astype(np.float64)
q_wxyz = d["root_quat"].astype(np.float64)
q_wxyz /= np.linalg.norm(q_wxyz, axis=1, keepdims=True)
q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
jp = d["joint_pos"].astype(np.float64)
assert jp.shape[1] == 29 and len(pos) == len(q_wxyz) == len(jp)
np.savetxt(dst, np.concatenate([pos, q_xyzw, jp], axis=1), delimiter=",", fmt="%.6f")
print(f"{src} -> {dst}: {len(jp)} frames, fps {float(d['fps'])}, duration {(len(jp) - 1) / float(d['fps']):.3f} s")
