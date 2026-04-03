# DROID Sim Evaluation

This repository contains scripts for evaluating DROID policies (and planners!) in a simple ISAAC Sim environment.

The simulator includes **5 scenes** (1–5), each with multiple variants that place objects in different configurations:

| Scene | Variants |
|-------|----------|
| 1     | 10 (0–9) |
| 2     | 10 (0–9) |
| 3     | 11 (0–8, 10–11)    |
| 4     | 10 (0–9) |
| 5     | 10 (0–9) |

The simulation is tuned to work *zero-shot* with DROID policies trained on the real-world DROID dataset, so no separate simulation data is required.

**Note:** The current simulator works best for policies trained with *joint position* action space (and *not* joint velocity control). We provide examples for evaluating pi0-FAST-DROID policies trained with joint position control below.

## Scenes

### Scene 1

> Example instruction: *"Put the Rubik's cube in the bowl."*

| Exterior | Wrist |
|----------|-------|
| ![Scene 1 exterior](docs/scene1_0_ext.png) | ![Scene 1 wrist](docs/scene1_0_wrist.png) |

---

### Scene 2

> Example instruction: *"Put the can in the mug."*

| Exterior | Wrist |
|----------|-------|
| ![Scene 2 exterior](docs/scene2_0_ext.png) | ![Scene 2 wrist](docs/scene2_0_wrist.png) |


---

### Scene 3

> Example instruction: *"Put the banana in the bin."*

| Exterior | Wrist |
|----------|-------|
| ![Scene 3 exterior](docs/scene3_0_ext.png) | ![Scene 3 wrist](docs/scene3_0_wrist.png) |


---

### Scene 4

> Example instruction: *"Put the cube on the mug and the cans in the bowl."*

A cluttered version of Scene 1 with many distractor objects (soup can, sardine tin, banana, mug, sugar box).

| Exterior | Wrist |
|----------|-------|
| ![Scene 4 exterior](docs/scene4_0_ext.png) | ![Scene 4 wrist](docs/scene4_0_wrist.png) |

---

### Scene 5

> Example instruction: *"Put 3 blocks in the bowl."*

A cluttered version of Scene 2 with multiple colored blocks as distractors.

| Exterior | Wrist |
|----------|-------|
| ![Scene 5 exterior](docs/scene5_0_ext.png) | ![Scene 5 wrist](docs/scene5_0_wrist.png) |

---

## Installation

Clone the repo
```bash
git clone --recurse-submodules git@github.com:tiptop-robot/droid-sim-evals.git
cd droid-sim-evals
```

Install uv (see: https://github.com/astral-sh/uv#installation)

For example (Linux/macOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate virtual environment
```bash
uv sync
source .venv/bin/activate
```

## Quick Start

First, make sure you download the simulation assets into the root of this directory
```bash
curl -O https://tiptop-sim-assets.s3.us-east-1.amazonaws.com/assets.zip 
unzip assets.zip
```

Then, in a separate terminal, launch the policy server on `localhost:8000`. 

For example, to launch a pi0.5 policy (with joint position control),
checkout [openpi](https://github.com/Physical-Intelligence/openpi) and use the `polaris` configs 
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

**Note**: We set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to avoid JAX hogging all the GPU memory (since Isaac Sim needs to use the same GPU).

Finally, run the evaluation script:
```bash
python tiptop_eval.py --scene <scene_id> --variant <variant_id> --instruction "<instruction>"
```

## Minimal Example

```python
env_cfg.set_scene(scene, variant)  # pass scene integer and variant integer
env = gym.make("DROID", cfg=env_cfg)

obs, _ = env.reset()
obs, _ = env.reset() # need second render cycle to get correctly loaded materials
client = # Your policy of choice

max_steps = env.env.max_episode_length
for _ in tqdm(range(max_steps), desc=f"Episode"):
    action = client.infer(obs, INSTRUCTION) # calling inference on your policy
    action = torch.tensor(ret["action"])[None]
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
env.close()
```
