import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import equinox as eqx
import imageio.v2 as imageio
import jax
from jax import numpy as jnp
import numpy as np
import pygame

from env.robotaxi import RobotaxiEnv
from utils.autoreset import AutoResetWrapper
from utils.renderer import PygameFrontend


class ActorCritic(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    actor_mean: eqx.nn.Linear
    actor_log_std: jax.Array
    critic_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key: jax.Array):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.fc1 = eqx.nn.Linear(obs_dim, hidden_dim, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.actor_mean = eqx.nn.Linear(hidden_dim, action_dim, key=k3)
        self.actor_log_std = jnp.full((action_dim,), -0.4, dtype=jnp.float32)
        self.critic_head = eqx.nn.Linear(hidden_dim, 1, key=k4)

    def __call__(self, x: jnp.ndarray):
        h = jax.nn.tanh(self.fc1(x))
        h = jax.nn.tanh(self.fc2(h))
        mean = self.actor_mean(h)
        log_std = jnp.clip(self.actor_log_std, -2.5, 0.5)
        value = jnp.squeeze(self.critic_head(h), axis=-1)
        return mean, log_std, value


def clean_obs(obs_vec: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(
        jnp.nan_to_num(obs_vec, nan=0.0, posinf=10.0, neginf=-10.0),
        -10.0,
        10.0,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Visualize trained PPO checkpoint on RobotaxiEnv")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ppo_robotaxi.eqx")
    p.add_argument("--map-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--save-frames", action="store_true", help="Save PNG frames to results/frames")
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument(
        "--record",
        type=str,
        default="",
        help="Path to save MP4 video directly, e.g. results/ppo_eval.mp4",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    key = jax.random.PRNGKey(0)
    key, init_key, model_key = jax.random.split(key, 3)

    env_params, init_state = RobotaxiEnv.init_params(
        key=init_key,
        map_id=args.map_id,
        max_steps=args.max_steps,
        discretization_scale=1,
        path_length=100,
    )
    init_obs_vec = clean_obs(RobotaxiEnv.get_observation(init_state, env_params).as_vector())
    obs_dim = int(init_obs_vec.shape[-1])
    action_dim = 2

    skeleton = ActorCritic(obs_dim, action_dim, args.hidden_dim, model_key)
    model = eqx.tree_deserialise_leaves(ckpt_path, skeleton)

    @jax.jit
    def policy_fn(obs_vec: jnp.ndarray) -> jnp.ndarray:
        mean, _, _ = model(clean_obs(obs_vec))
        return jnp.tanh(mean)

    debug_counter = [0]

    def agent_fn(state):
        obs = RobotaxiEnv.get_observation(state, env_params)
        obs_vec = obs.as_vector()
        action = policy_fn(obs_vec)

        if debug_counter[0] % 30 == 0:
            print(
                f"\nStep: {debug_counter[0]}"
                f"\n  Position: {state.agent_pos}"
                f"\n  Goal: {state.goal_pos}"
                f"\n  Heading: {state.heading:.3f} rad"
                f"\n  Speed: {state.speed:.3f}"
                f"\n  Steering: {state.steering:.3f}"
                f"\n  Goal local: {obs.goal_vec_local}"
                f"\n  Goal turn sin/cos: "
                f"{obs.turn_sin_goal:.3f}, {obs.turn_cos_goal:.3f}"
                f"\n  Path direction: {obs.direction_of_path}"
                f"\n  Path turn sin/cos: "
                f"{obs.turn_sin_path:.3f}, {obs.turn_cos_path:.3f}"
                f"\n  Action [throttle, steer]: {action}"
            )

        debug_counter[0] += 1
        return action

    env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)
    frontend = PygameFrontend(env, env_params, init_state, eval_mode=True, agent_fn=agent_fn)

    if args.record:
        out_path = Path(args.record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_path), fps=30)
        saved = 0
        try:
            while frontend.running and saved < args.max_frames:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        frontend.running = False

                action = agent_fn(frontend.state)
                frontend.obs, frontend.state, _, _, frontend.info = frontend.env.step(
                    frontend.key, frontend.state, action, frontend.params
                )
                frontend.draw()

                frame = np.transpose(
                    pygame.surfarray.array3d(frontend.screen), (1, 0, 2)
                )
                writer.append_data(frame)
                saved += 1
                frontend.clock.tick(60)
        finally:
            writer.close()
            pygame.quit()
        print(f"Saved video ({saved} frames) to {out_path.resolve()}")
        return

    if not args.save_frames:
        frontend.run()
        return

    frames_dir = Path("results") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    while frontend.running and saved < args.max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                frontend.running = False

        action = agent_fn(frontend.state)
        frontend.obs, frontend.state, _, _, frontend.info = frontend.env.step(
            frontend.key, frontend.state, action, frontend.params
        )
        frontend.draw()
        pygame.image.save(frontend.screen, str(frames_dir / f"frame_{saved:04d}.png"))
        saved += 1
        frontend.clock.tick(60)

    pygame.quit()
    print(f"Saved {saved} frames to {frames_dir.resolve()}")


if __name__ == "__main__":
    main()