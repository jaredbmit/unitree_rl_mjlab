# Single-clip tracking on the G1

Task `Unitree-G1-Tracking-No-State-Estimation`, trained on one walking clip at a time with the unmodified upstream
config (1425b15), exported for the C++ deployer and run on a G1 29-DoF. Besides the files below, the fork adds one
deployer fix (c6fb3b2): `State_Mimic.cpp` kept an Eigen view of a temporary, so the anchor-orientation observation read
freed memory; in our g++-9 -O3 aarch64 build the robot froze at Mimic entry. Upstream PR #56 fixes the same line.

## Install
```
uv venv --python 3.11 && uv pip install -e . --exclude-newer 2026-04-14
```
Without the date cutoff uv resolves mujoco 3.17, which breaks mujoco-warp 3.5.0 (`mjtEnableBit has no
mjENBL_MULTICCD`). Exact versions used: `requirements-pinned.txt` (mjlab 1.2.0, mujoco 3.6.0, torch 2.11.0, rsl-rl-lib 5.0.1).

## Motion
The CSV format (LAFAN1 retargeting) is one row per frame: root x,y,z, root quat x,y,z,w, 29 joint positions in
Unitree SDK order. `csv_to_npz.py` resamples to 50 fps and stores mjlab FK of all bodies; it writes into
`src/assets/motions/g1/` relative to the cwd. CPU works (`--device cpu`, about 2 min of kernel compiles).
```
python scripts/csv_to_npz.py --input-file src/assets/motions/g1/walk1_subject1_0-12s.csv \
  --output-name lafan_walk1_subject1_0-12s.npz --input-fps 30 --output-fps 50 --robot g1 --device cpu
```
`--line-range` does not parse with tyro here, so clips are cut with `head`. A root-pose npz (root_pos, root_quat wxyz,
joint_pos, fps; our BONES-SEED clip) is first converted to that CSV without resampling (`--input-fps` = its fps):
```
python scripts/g1deploy_npz_to_csv.py clip.npz clip.csv
python scripts/check_motion_npz.py <source.csv|source.npz> <converted.npz> [--source-fps 30] [--plot out.png]
```

LAFAN clip: `walk1_subject1_0-12s.csv` = `head -n 361 g1/walk1_subject1.csv` (0 to 12 s at 30 fps, CRLF kept; source
7840 rows, sha256 68c7887438e94f1f3d9ebacd3a8ce59251a22c7b05129a0f5d7cacaedc48c28d). Unitree's HF dataset returned 404
on 2026-11-02, so the source came from the mirror `lvhaidong/LAFAN1_Retargeting_Dataset@ce15729`.
LAFAN1 is CC BY-NC-ND 4.0 (Ubisoft La Forge); the G1 retarget is Unitree's LAFAN1_Retargeting_Dataset.

## Train
```
sbatch -J lafan12 scripts/train_tracking.sbatch lafan_walk1_subject1_0-12s src/assets/motions/g1/lafan_walk1_subject1_0-12s.npz
```
Defaults: 4096 envs, 30001 iterations, output in `logs/` (or `OUT=<dir>`). It also runs with `bash`; resubmitting the
same command resumes from the newest checkpoint of that run name. It runs:
```
WANDB_MODE=disabled python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion-file <motion.npz> \
  --env.scene.num-envs 4096 --agent.logger tensorboard --agent.run-name <run name> --agent.max-iterations 30001
```
Viewer: `python scripts/play.py Unitree-G1-Tracking-No-State-Estimation --motion_file=<motion.npz> --checkpoint_file=<model_N.pt>`.

## Export and package
Training saves write `policy.onnx` (normalizer included) next to `model_<it>.pt`. The deployer also needs
`params/deploy.yaml` and the motion npz; nothing upstream writes deploy.yaml:
```
python scripts/make_deploy_package.py <model_N.pt | run name> <motion.npz> deploy/robots/g1/config/policy/mimic/<state>
```
It re-exports the ONNX files and writes deploy.yaml from the shipped dance one with the gains, default pose, action
scale and offset of the training env. The shipped dance deploy.yaml has another default pose (HOME, see Results).

## Deploy
Build and run `deploy/robots/g1` as in README section 4. g1_ctrl loads `config/policy/mimic/<state>/` for every Mimic
state in `deploy/robots/g1/config/config.yaml` (and fails at startup if one is missing). A new state needs an entry
under `FSM._`, a transition into it and a state block:
```
FSM:
  _:
    Mimic_Lafan_walk1_0_12s: {id: 7, type: Mimic}
  Velocity:
    transitions:
      Mimic_Lafan_walk1_0_12s: RB + Y.on_pressed
  Mimic_Lafan_walk1_0_12s:
    transitions: {Passive: LT + B.on_pressed, Velocity: RT + A.on_pressed}
    motion_file: config/policy/mimic/lafan_walk1_0_12s/params/lafan_walk1_subject1_0-12s.npz
    policy_dir: config/policy/mimic/lafan_walk1_0_12s/
```
Sequence on the robot: hanging, L2+R2 (debug mode); `./g1_ctrl --network=eth0`; L2+Up (FixStand); lower until the
feet carry weight; R2+A (Velocity, shipped velocity/v0); R1+Y (Mimic). Entry aligns the clip's yaw to the robot, not
its position. At the clip end it returns to Velocity. L2+B is Passive; there is no automatic fall detection.

## Files of the LAFAN 0-12 s run
- `src/assets/motions/g1/`: `walk1_subject1.csv` (as downloaded), `walk1_subject1_0-12s.csv` (the cut clip, from
  `head -n 361 walk1_subject1.csv`) and `lafan_walk1_subject1_0-12s.npz` (its csv_to_npz output, 600 frames).
- `results/lafan_walk1_subject1_0-12s/`: `model_30000.pt` (final), `policy.onnx`, the TensorBoard events file (one run
  dir, no resume), `params/{env,agent}.yaml` from train.py with env.yaml's absolute `motion_file` made repo-relative.
- `deploy/robots/g1/config/policy/mimic/lafan_walk1_0_12s/`: package of model_30000, regenerated with
  make_deploy_package.py; identical to the one run on the robot except for the recorded checkpoint path.

## Results
Three runs, one seed (42) each, public config, 4096 envs, 24 steps per env per iteration, one Titan RTX per run
(2.0 to 2.6 s per iteration).

| run | clip | iterations | wall time |
|---|---|---|---|
| lafan_walk1_subject1_0-12s | LAFAN walk1_subject1 0-12 s: stands 3.5 s, walks 3.56 m straight, stops | 30000 | 19 h 38 min |
| ours_loco_walk_slow_short | BONES-SEED walk, 4.46 s, 2.43 m, 4 long strides (not in this repo for licensing reasons) | 30000 | 18 h 57 min |
| lafan_walk1_subject1 | LAFAN walk1_subject1 full, 261 s wandering walk with turns | cancelled at 29738, last checkpoint 29500 | 21 h 45 min |

Final training reward (100-iteration mean, lafan 0-12 s / ours / lafan full): 42.24 / 38.89 / 39.12; failures ended
0.02 / 0.11 / 0.13 % of episodes.

mjlab play of the final checkpoints, 16 envs from clip frame 0 once through the clip (play mode: no observation noise,
pushes or start randomization; startup DR and reset joint noise on):

| | lafan 0-12 s @30000 | ours @30000 | lafan full @29500 |
|---|---|---|---|
| envs that fell | 0 / 16 | 0 / 16 | 0 / 16 |
| legs / arms / waist joint rms (rad) | 0.062 / 0.080 / 0.047 | 0.091 / 0.075 / 0.066 | 0.083 / 0.097 / 0.054 |
| pelvis pitch error, mean (rad) | +0.059 | +0.090 | +0.034 |
| pelvis start-to-end distance, robot / clip (m) | 3.65 / 3.56 | 2.42 / 2.41 | 1.70 / 1.74 |
| swing clearance on long steps, robot / clip (m) | 0.029 / 0.022 | 0.037 / 0.074 | 0.038 / 0.029 |

Real G1 (robot 2, 2026-11-02, gantry slack, g1_ctrl with c6fb3b2): three plays of each short clip, entered from
standing under velocity/v0. Joint rms over clip time 1.5 s to the end (the first 1.5 s, where the arms swing out to the
clip's opening pose, are left out). sim2sim = the same packages in our MuJoCo sim2sim harness (not in this repo).
Earlier plays that evening hit a right-arm motor fault (cleared by a power cycle) and are not used. The full LAFAN
run was not tested on the robot.

| | lafan 0-12 s sim2sim | lafan 0-12 s robot | ours sim2sim | ours robot |
|---|---|---|---|---|
| legs rms (rad) | 0.069 | 0.144 to 0.174 | 0.096 | 0.165 to 0.192 |
| arms rms (rad) | 0.076 | 0.245 to 0.269 | 0.078 | 0.222 to 0.237 |
| waist rms (rad) | 0.047 | 0.060 | 0.065 | 0.109 to 0.116 |
| pelvis pitch error, mean (deg) | +4.3 | +2.5 to +3.1 | +5.0 | +8.7 to +10.4 |
| max tilt, clip in brackets (deg) | 12 (7) | 15 to 18 (7) | 11 (8) | 21 to 24 (8) |

On the robot the lafan arm error is mostly the wrist rolls, held at about +-0.35 rad where the reference is +-0.81.
Unitree's shipped dance policy on the same robot: legs rms 0.128 to 0.130 rad against 0.111 in sim2sim (ratio about
1.15); the ratio is 2.1 to 2.5 for lafan 0-12 s and 1.7 to 2.0 for ours.

The public config at 1425b15 is not the config that produced Unitree's shipped policies. Those were trained with the
repo's earlier vendored mjlab (cbb1ade to e3bf14c); commit 4a734a8 switched to pip mjlab. Differences in behaviour:
- default pose and action offset: HOME (knee 0.3, hip pitch -0.1, ankle pitch -0.2, shoulder pitch 0.35, elbow 0.87)
  then, KNEES_BENT (0.669, -0.312, -0.363, 0.2, 0.6) now; the shipped ONNX normalizer statistics show HOME;
- anchor-height and end-effector-height terminations: 0.5 m then, 0.25 m now;
- foot friction 0.3 to 1.2 per foot capsule then, one value per env now; torso COM x offset +-0.025 then, +-0.05 m now;
- self-collision penalty: contact count at the last substep then, substeps with a contact above 10 N now;
- rsl_rl 3.1.0 then, 5.0.1 now (same hyperparameters; gradient clipping per network instead of joint); torch 2.9.1
  then, 2.11.0 now.

Retraining the shipped dance clip (dance1_subject2, 131.5 s) with the public config and the setup above for 30000
iterations: in our MuJoCo sim2sim over the first 25 s, legs / arms / waist rms 0.119 / 0.147 / 0.063 rad against
0.111 / 0.134 / 0.055 for the shipped policy (within 7, 10 and 15 %), pelvis pitch rms 4.2 against 4.1 deg, no fall.
Its training reward was still rising at 30000 (27.5 at 20000, 29.4 at 30000).
