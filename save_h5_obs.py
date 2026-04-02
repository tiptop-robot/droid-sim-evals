"""Save an initial observation from a DROID sim scene to an H5 file."""
import argparse
import logging
import h5py
import torch
import gymnasium as gym
import tyro

from src.sim_evals.sim_utils import settle_sim

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(scene: int = 1, variant: int = 0, output: str = "observation.h5", headless: bool = True):
    """Capture and save an initial observation from the simulator.

    Args:
        scene: Scene number (1-5).
        variant: Scene variant (0-9).
        output: Path to save the H5 file.
        headless: Run without GUI. Set to False to open the Isaac Sim viewport.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(scene), variant)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second reset for correct material loading

    obs = settle_sim(env, obs, steps=100)

    def to_numpy(t):
        return t[0].cpu().numpy() if isinstance(t, torch.Tensor) else t[0]

    p = obs["policy"]
    with h5py.File(output, "w") as f:
        f.create_dataset("rgb",              data=to_numpy(p["wrist_cam"]))
        f.create_dataset("depth",            data=to_numpy(p["wrist_depth"]))
        f.create_dataset("intrinsic_matrix", data=to_numpy(p["wrist_intrinsics"]))
        f.create_dataset("pos_w",            data=to_numpy(p["wrist_cam_pos_w"]))
        f.create_dataset("quat_w_ros",       data=to_numpy(p["wrist_cam_quat_w"]))  # [w, x, y, z]
        f.create_dataset("q_init",           data=to_numpy(p["arm_joint_pos"]))

    logger.info(f"Saved observation to {output}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    tyro.cli(main)