"""Evaluation script that uses the tiptop websocket server for planning.

This script connects to a running tiptop websocket server, sends initial
observations (RGB, depth, camera params, task instruction), receives a
trajectory plan, and executes it in the Isaac Sim environment.

Usage:
    # First, start the tiptop websocket server:
    # (in tiptop-robot) pixi run python -m tiptop.websocket_server --port 8765

    # Then run this evaluation:
    uv run python tiptop_ws_eval.py --scene 1 --ws-host localhost --ws-port 8765
"""

import argparse
import logging
from datetime import datetime
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

import cv2
import gymnasium as gym
import mediapy
import torch
import tyro
from tqdm import tqdm

from src.sim_evals.inference.tiptop_websocket import TiptopWebsocketClient
from src.visual_utils import add_top_padding, overlay_timer_ms


def main(
    instruction: str,
    episodes: int = 1,
    headless: bool = True,
    scene: int = 1,
    variant: int = 0,
    ws_host: str = "localhost",
    ws_port: int = 8765,
):
    """Run evaluation using tiptop websocket server.

    Args:
        episodes: Number of episodes to run
        headless: If True (default), runs without the Isaac Sim GUI and only saves a video file.
            Set to False to open the Isaac Sim viewport for live visualization:
            ``uv run python tiptop_eval.py --headless False``
        scene: Scene number (1-5)
        variant: Scene variant (0-9)
        ws_host: Tiptop websocket server host
        ws_port: Tiptop websocket server port
        instruction: Natural language task instruction.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Tiptop websocket evaluation")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import src.sim_evals.environments  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )

    env_cfg.set_scene(str(scene))
    env_cfg.episode_length_s = 90.0
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()  # Need second render cycle to get correctly loaded materials

    # Connect to tiptop websocket server
    logger.info(f"Connecting to tiptop server at ws://{ws_host}:{ws_port}...")
    client = TiptopWebsocketClient(host=ws_host, port=ws_port)

    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    max_steps = env.env.max_episode_length
    video_fps = 15

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            frame_idx = 0
            # run sim for ~1 second so objects settle into place
            settle_steps = 15
            for _ in range(settle_steps):
                hold_action = torch.cat([
                    obs["policy"]["arm_joint_pos"],
                    obs["policy"]["gripper_pos"],
                ], dim=-1)
                obs, _, _, _, _ = env.step(hold_action)
            env.env.episode_length_buf[:] = 0
            plan_failed = False
            for i in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                try:
                    ret = client.infer(obs, instruction)
                except Exception as e:
                    logger.error(f"Planning failed for episode {ep+1}: {e}. Skipping.")
                    plan_failed = True
                    break

                if client.plan_done:
                    logger.info(f"Plan fully executed at step {frame_idx}")
                    break

                viz = np.concatenate([ret["right_image"], ret["wrist_image"]], axis=1)
                viz = add_top_padding(viz, pad_px=40)
                elapsed_ms = int(frame_idx * 1000 / video_fps)
                overlay_timer_ms(viz, elapsed_ms)
                if not headless:
                    try:
                        cv2.imshow("Camera View", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    except cv2.error:
                        pass

                video.append(viz)
                frame_idx += 1

                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            client.reset()
            if plan_failed:
                video = []
                continue
            mediapy.write_video(
                video_dir / f"tiptop_scene{scene}_ep{ep}.mp4",
                video,
                fps=video_fps,
            )
            video = []
            logger.info(f"Saved video to {video_dir / f'tiptop_scene{scene}_ep{ep}.mp4'}")

    client.close()
    env.close()
    simulation_app.close()

    video_lines = [
        f"  Instruction : {instruction}",
        f"  Scene       : {scene}",
        f"  Episodes    : {episodes}",
        f"  Output dir  : {video_dir.resolve()}",
    ]
    for ep in range(episodes):
        video_path = video_dir / f"tiptop_scene{scene}_ep{ep}.mp4"
        if video_path.exists():
            video_lines.append(f"  Video ep{ep}   : {video_path.resolve()}")
    logger.info("Run complete\n" + "\n".join(video_lines))


if __name__ == "__main__":
    tyro.cli(main)
