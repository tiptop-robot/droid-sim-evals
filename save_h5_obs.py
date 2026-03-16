"""Save an initial observation from a DROID sim scene to an H5 file."""
import argparse
import logging
import h5py
import torch
import gymnasium as gym
import tyro

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main(scene: int = 1, output: str = "observation.h5"):
    """Capture and save an initial observation from the simulator.

    Args:
        scene: Scene number (1, 2, or 3).
        output: Path to save the H5 file.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg("DROID", device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.set_scene(str(scene))
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # second reset for correct material loading

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