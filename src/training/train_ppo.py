import argparse
import csv
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
from jax import numpy as jnp
import optax

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env.robotaxi import RobotaxiEnv


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


class Transition(NamedTuple):
    obs: jnp.ndarray
    raw_action: jnp.ndarray
    log_prob: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray


def clean_obs(obs_vec: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(
        jnp.nan_to_num(obs_vec, nan=0.0, posinf=10.0, neginf=-10.0),
        -10.0,
        10.0,
    )


def sample_action(model: ActorCritic, obs: jnp.ndarray, key: jax.Array):
    mean, log_std, value = model(obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, shape=mean.shape)
    raw_action = mean + std * eps
    action = jnp.tanh(raw_action)

    gauss_lp = -0.5 * (
        ((raw_action - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
    )
    tanh_corr = jnp.log(1.0 - action**2 + 1e-6)
    log_prob = jnp.sum(gauss_lp - tanh_corr, axis=-1)
    return action, raw_action, log_prob, value


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    def scan_fn(carry, inp):
        gae, next_val = carry
        rew, val, done = inp
        not_done = 1.0 - done.astype(jnp.float32)
        delta = rew + gamma * next_val * not_done - val
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, val), gae

    _, advantages = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on RobotaxiEnv")
    p.add_argument("--map-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-length", type=int, default=256)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    key, init_key, model_key = jax.random.split(key, 3)

    env_params, init_state = RobotaxiEnv.init_params(
        key=init_key,
        map_id=args.map_id,
        max_steps=args.max_steps,
        discretization_scale=1,
        path_length=100,
    )
    init_obs_struct = RobotaxiEnv.get_observation(init_state, env_params)
    init_obs_vec = clean_obs(init_obs_struct.as_vector())
    obs_dim = int(init_obs_vec.shape[-1])
    action_dim = 2

    states = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.asarray(x)[None, ...], args.num_envs, axis=0),
        init_state,
    )
    obs_vec = jnp.repeat(init_obs_vec[None, ...], args.num_envs, axis=0)
    ep_returns = jnp.zeros((args.num_envs,), dtype=jnp.float32)
    ep_lengths = jnp.zeros((args.num_envs,), dtype=jnp.float32)

    model = ActorCritic(obs_dim, action_dim, args.hidden_dim, model_key)
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(0.5),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @jax.jit
    def rollout_and_update(model, opt_state, states, obs_vec, ep_ret, ep_len, rng):
        env_keys = jax.random.split(rng, args.num_envs)

        def step_fn(carry, _):
            cur_states, cur_obs, cur_ret, cur_len, cur_rng = carry
            cur_rng, act_rng = jax.random.split(cur_rng)

            act_keys = jax.random.split(act_rng, args.num_envs)
            actions, raw_actions, log_probs, values = jax.vmap(
                lambda o, k: sample_action(model, o, k)
            )(cur_obs, act_keys)

            next_obs_struct, next_states, rewards, dones, infos = jax.vmap(
                lambda st, act, ek: RobotaxiEnv.step(ek, st, act, env_params)
            )(cur_states, actions, env_keys)
            next_obs_vec = jax.vmap(lambda o: clean_obs(o.as_vector()))(next_obs_struct)

            updated_ret = cur_ret + rewards
            updated_len = cur_len + 1.0

            completed_ret = jnp.where(dones, updated_ret, jnp.nan)
            completed_len = jnp.where(dones, updated_len, jnp.nan)

            reset_states = jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.asarray(x)[None, ...], args.num_envs, axis=0),
                init_state,
            )
            reset_obs = jnp.repeat(init_obs_vec[None, ...], args.num_envs, axis=0)

            final_states = jax.tree_util.tree_map(
                lambda ns, rs: jnp.where(
                    dones.reshape((-1,) + (1,) * (ns.ndim - 1)), rs, ns
                ),
                next_states,
                reset_states,
            )
            final_obs = jnp.where(dones[:, None], reset_obs, next_obs_vec)
            final_ret = jnp.where(dones, 0.0, updated_ret)
            final_len = jnp.where(dones, 0.0, updated_len)

            tr = Transition(
                obs=cur_obs,
                raw_action=raw_actions,
                log_prob=log_probs,
                reward=rewards,
                done=dones,
                value=values,
            )
            aux = (completed_ret, completed_len, infos["collision"], infos["success"])
            return (final_states, final_obs, final_ret, final_len, cur_rng), (tr, aux)

        rng, roll_rng = jax.random.split(rng)
        (last_states, last_obs, last_ret, last_len, _), (traj, aux) = jax.lax.scan(
            step_fn, (states, obs_vec, ep_ret, ep_len, roll_rng), None, length=args.rollout_length
        )

        _, _, last_values = jax.vmap(model)(last_obs)
        advantages, returns = jax.vmap(
            lambda r, v, d, lv: compute_gae(r, v, d, lv, args.gamma, args.gae_lambda),
            in_axes=(1, 1, 1, 0),
            out_axes=(1, 1),
        )(traj.reward, traj.value, traj.done, last_values)

        total_samples = args.num_envs * args.rollout_length
        mb_size = total_samples // args.minibatches

        flat_obs = traj.obs.reshape((-1, obs_dim))
        flat_raw = traj.raw_action.reshape((-1, action_dim))
        flat_lp = traj.log_prob.reshape((-1,))
        flat_adv = advantages.reshape((-1,))
        flat_ret = returns.reshape((-1,))
        flat_adv = (flat_adv - jnp.mean(flat_adv)) / (jnp.std(flat_adv) + 1e-8)

        def loss_fn(m, bo, br, blp, badv, bret):
            mean, log_std, value = jax.vmap(m)(bo)
            std = jnp.exp(log_std)
            action = jnp.tanh(br)
            gauss_lp = -0.5 * (
                ((br - mean) / std) ** 2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi)
            )
            tanh_corr = jnp.log(1.0 - action**2 + 1e-6)
            new_lp = jnp.sum(gauss_lp - tanh_corr, axis=-1)

            ratio = jnp.exp(new_lp - blp)
            surr1 = ratio * badv
            surr2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * badv
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
            value_loss = 0.5 * jnp.mean((value - bret) ** 2)
            entropy = jnp.mean(
                jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e) + tanh_corr, axis=-1)
            )
            return policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

        def epoch_fn(carry, _):
            m, opt_st, ep_rng = carry
            ep_rng, perm_rng = jax.random.split(ep_rng)
            perm = jax.random.permutation(perm_rng, total_samples)[: mb_size * args.minibatches]

            mb_o = flat_obs[perm].reshape((args.minibatches, mb_size, obs_dim))
            mb_r = flat_raw[perm].reshape((args.minibatches, mb_size, action_dim))
            mb_lp = flat_lp[perm].reshape((args.minibatches, mb_size))
            mb_adv = flat_adv[perm].reshape((args.minibatches, mb_size))
            mb_ret = flat_ret[perm].reshape((args.minibatches, mb_size))

            def mb_fn(mb_carry, batch):
                mm, oo = mb_carry
                bo, br, blp, badv, bret = batch
                grads = eqx.filter_grad(loss_fn)(mm, bo, br, blp, badv, bret)
                updates, oo = optimizer.update(
                    grads, oo, eqx.filter(mm, eqx.is_inexact_array)
                )
                mm = eqx.apply_updates(mm, updates)
                return (mm, oo), None

            (m, opt_st), _ = jax.lax.scan(
                mb_fn, (m, opt_st), (mb_o, mb_r, mb_lp, mb_adv, mb_ret)
            )
            return (m, opt_st, ep_rng), None

        rng, ep_rng = jax.random.split(rng)
        (model, opt_state, _), _ = jax.lax.scan(
            epoch_fn, (model, opt_state, ep_rng), None, length=args.epochs
        )

        comp_ret, comp_len, colls, succs = aux
        valid_ret = jnp.isfinite(comp_ret)
        mean_ep_ret = jnp.where(
            jnp.sum(valid_ret) > 0,
            jnp.sum(jnp.where(valid_ret, comp_ret, 0.0)) / jnp.maximum(jnp.sum(valid_ret), 1),
            jnp.mean(traj.reward) * args.rollout_length,
        )
        valid_len = jnp.isfinite(comp_len)
        mean_ep_len = jnp.where(
            jnp.sum(valid_len) > 0,
            jnp.sum(jnp.where(valid_len, comp_len, 0.0)) / jnp.maximum(jnp.sum(valid_len), 1),
            jnp.asarray(args.rollout_length, dtype=jnp.float32),
        )

        metrics = {
            "mean_step_reward": jnp.mean(traj.reward),
            "ep_return": mean_ep_ret,
            "ep_length": mean_ep_len,
            "collision_rate": jnp.mean(colls.astype(jnp.float32)),
            "success_rate": jnp.mean(succs.astype(jnp.float32)),
        }
        return model, opt_state, last_states, last_obs, last_ret, last_len, rng, metrics

    ckpt_dir = Path("checkpoints")
    res_dir = Path("results")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    log_path = res_dir / "ppo_training.csv"
    ckpt_path = ckpt_dir / "ppo_robotaxi.eqx"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "update", "steps", "mean_step_reward", "ep_return",
            "ep_length", "collision_rate", "success_rate",
        ])

        for upd in range(1, args.updates + 1):
            model, opt_state, states, obs_vec, ep_returns, ep_lengths, key, m = (
                rollout_and_update(
                    model, opt_state, states, obs_vec, ep_returns, ep_lengths, key
                )
            )
            total_steps = upd * args.num_envs * args.rollout_length
            if upd % 25 == 0 or upd == 1 or upd == args.updates:
                row = [
                    upd,
                    total_steps,
                    float(m["mean_step_reward"]),
                    float(m["ep_return"]),
                    float(m["ep_length"]),
                    float(m["collision_rate"]),
                    float(m["success_rate"]),
                ]
                writer.writerow(row)
                print(
                    f"upd={upd:4d}/{args.updates} steps={total_steps:7d} "
                    f"step_rew={row[2]:+.4f} ep_ret={row[3]:+.2f} "
                    f"ep_len={row[4]:.1f} coll={row[5]:.3f} succ={row[6]:.3f}"
                )

    eqx.tree_serialise_leaves(ckpt_path, model)
    print(f"Training completed! Checkpoint: {ckpt_path} | Log: {log_path}")


if __name__ == "__main__":
    main()