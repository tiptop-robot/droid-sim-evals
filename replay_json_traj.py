import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gymnasium as gym
import mediapy
import numpy as np
import torch
import tyro
from tqdm import tqdm
from openpi_client import image_tools


def load_plan_from_json(json_path: str) -> dict:
    """Load a serialized TiPToP plan from a JSON file.

    Expected format (from tiptop planning.serialize_plan):
        {"version": "...", "q_init": [...], "steps": [...]}
    Each step is either:
        {"type": "trajectory", "positions": [[...], ...], ...}
        {"type": "gripper", "action": "open"|"close", ...}
    """
    with open(json_path) as f:
        plan = json.load(f)
    plan["q_init"] = np.array(plan["q_init"], dtype=np.float32)
    for step in plan["steps"]:
        if step["type"] == "trajectory":
            step["positions"] = np.array(step["positions"], dtype=np.float32)
            if "velocities" in step:
                step["velocities"] = np.array(step["velocities"], dtype=np.float32)
    return plan


class LocalPlanClient:
    """Step through a cuTAMP plan using TiptopWebsocketClient-style execution."""

    def __init__(
        self,
        plan: dict,
        gripper_action_steps: int = 20,
        sim_control_hz: float = 15.0,
        curobo_interp_hz: float = 50.0,
        q_init_tol: float = 0.02,
    ) -> None:
        self._plan = plan["steps"]
        self._q_init = np.asarray(plan["q_init"], dtype=np.float32) if "q_init" in plan else None
        self._gripper_action_steps = gripper_action_steps
        self._waypoint_stride = max(1, int(round(curobo_interp_hz / sim_control_hz)))
        self._q_init_tol = q_init_tol

        self._current_plan_step = 0
        self._current_trajectory: Optional[np.ndarray] = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending: Optional[str] = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0
        self._q_init_reached = self._q_init is None

    def reset(self) -> None:
        self._current_plan_step = 0
        self._current_trajectory = None
        self._current_waypoint_idx = 0
        self._gripper_action_pending = None
        self._gripper_action_steps_remaining = 0
        self._last_gripper_state = 0.0
        self._q_init_reached = self._q_init is None

    def infer(self, obs: dict) -> dict:
        curr_obs = self._extract_observation(obs)
        return self._step_plan(curr_obs)

    def _step_plan(self, curr_obs: dict) -> dict:
        if not self._q_init_reached:
            q_curr = curr_obs["joint_position"].flatten()
            if self._q_init is not None and q_curr.shape == self._q_init.shape:
                dist = np.linalg.norm(q_curr - self._q_init)
                if dist <= self._q_init_tol:
                    self._q_init_reached = True
                else:
                    action = np.concatenate([self._q_init, np.array([self._last_gripper_state])])
                    return self._make_result(action, curr_obs)
            else:
                self._q_init_reached = True

        if self._gripper_action_pending is not None:
            if self._gripper_action_steps_remaining > 0:
                self._gripper_action_steps_remaining -= 1
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._last_gripper_state = gripper_val
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)
            else:
                self._last_gripper_state = 1.0 if self._gripper_action_pending == "close" else 0.0
                self._gripper_action_pending = None
                self._current_plan_step += 1

        if self._current_trajectory is None or self._current_waypoint_idx >= len(self._current_trajectory):
            if self._plan is None or self._current_plan_step >= len(self._plan):
                joint_pos = curr_obs["joint_position"]
                gripper_val = (
                    float(curr_obs["gripper_position"][0])
                    if len(curr_obs["gripper_position"]) > 0
                    else self._last_gripper_state
                )
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)

            step = self._plan[self._current_plan_step]
            if step.get("type") == "gripper":
                action = step["action"]
                if action not in {"open", "close"}:
                    raise ValueError(f"Unknown gripper action: {action}")
                self._gripper_action_pending = action
                self._gripper_action_steps_remaining = self._gripper_action_steps
                joint_pos = curr_obs["joint_position"]
                gripper_val = 1.0 if action == "close" else 0.0
                self._last_gripper_state = gripper_val
                action = np.concatenate([joint_pos.flatten(), np.array([gripper_val])])
                return self._make_result(action, curr_obs)

            if "positions" in step:
                full_trajectory = np.asarray(step["positions"], dtype=np.float32)
            elif "plan" in step:
                full_trajectory = step["plan"].position.cpu().numpy()
            else:
                # Skip metadata or unsupported steps.
                self._current_plan_step += 1
                return self._step_plan(curr_obs)
            self._current_trajectory = self._subsample_trajectory(full_trajectory)
            self._current_waypoint_idx = 0
            self._current_plan_step += 1

        waypoint = self._current_trajectory[self._current_waypoint_idx]
        self._current_waypoint_idx += 1
        if waypoint.shape[0] == 7:
            action = np.concatenate([waypoint, np.array([self._last_gripper_state])])
        else:
            action = waypoint
        return self._make_result(action, curr_obs)

    def _subsample_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        if self._waypoint_stride <= 1 or len(trajectory) == 0:
            return trajectory
        indices = np.arange(0, len(trajectory), self._waypoint_stride)
        if indices[-1] != len(trajectory) - 1:
            indices = np.append(indices, len(trajectory) - 1)
        return trajectory[indices]

    def _make_result(self, action: np.ndarray, curr_obs: dict) -> dict:
        if image_tools is not None:
            img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
            img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            viz = np.concatenate([img1, img2], axis=1)
        else:
            viz = curr_obs["wrist_image"]
        return {
            "action": action,
            "viz": viz,
            "right_image": curr_obs["right_image"],
            "wrist_image": curr_obs["wrist_image"],
        }

    def _extract_observation(self, obs_dict: dict) -> dict:
        policy = obs_dict["policy"]
        return {
            "right_image": policy["external_cam"][0].clone().detach().cpu().numpy(),
            "wrist_image": policy["wrist_cam"][0].clone().detach().cpu().numpy(),
            "joint_position": policy["arm_joint_pos"].clone().detach().cpu().numpy(),
            "gripper_position": policy["gripper_pos"].clone().detach().cpu().numpy(),
        }

def main(
        json_path: str,
        episodes: int = 1,
        headless: bool = True,
        scene: int = 1,
        ):
    """Run evaluation using a local cuTAMP plan JSON file.

    Args:
        json_path: Path to cuTAMP plan JSON file (from tiptop planning.serialize_plan).
        episodes: Number of episodes to run.
        headless: If True (default), runs without the Isaac Sim GUI and only saves a video file.
            Set to False to open the Isaac Sim viewport for live visualization:
            ``uv run python replay_json_traj.py --headless False``
        scene: Scene number (1-5).
    """
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.set_scene(scene)
    env_cfg.episode_length_s = 30.0 # LENGTH OF EPISODE
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials

    cutamp_plan = load_plan_from_json(json_path)
    client = LocalPlanClient(cutamp_plan, gripper_action_steps=20, sim_control_hz=15.0, curobo_interp_hz=50.0)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            # Settle phase: run sim for ~1 second so objects settle into place
            settle_steps = 15  # 15 steps at 15 Hz = 1 second
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.episode_length_buf[:] = 0  # don't count settle steps toward episode length
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                ret = client.infer(obs)
                if not headless:
                    try:
                        cv2.imshow("Camera View", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    except cv2.error:
                        pass
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)

                if term or trunc:
                    break

            client.reset()
            video_path = video_dir / f"cutamp_scene{scene}_ep{ep}.mp4"
            mediapy.write_video(video_path, video, fps=15)
            logger.info(f"Saved video to {video_path}")
            video = []

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)

    
