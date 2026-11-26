from functools import partial
from typing import Tuple, Dict, Any

import chex
import equinox
import jax
from jax import numpy as jnp

from env.base import (
    BaseEnv,
    BaseEnvParams,
    NUM_RAY_SENSORS,
    FOV,
    FPS,
    OBSTACLE_COST,
    RADIUS,
)
from utils import utils


ROBOT_RADIUS: float = 0.65              
OBSTACLE_RADIUS: float = RADIUS
ROBOT_COLLISION_MARGIN: float = 1.0     
GRID_INFLATE_RADIUS: float = 1.15       
REPLAN_EVERY_N_STEPS: int = 10            # перестраивать маршрут каждые 6 шагов

MAX_SPEED: float = 2.5
MIN_SPEED: float = 0.0
MAX_ACCEL: float = 2.0
MAX_BRAKE: float = 3.0
MAX_STEER_ANGLE: float = 0.55            
WHEELBASE: float = 1.4

LINEAR_DRAG: float = 0.20
ROLLING_RESISTANCE: float = 0.25

GOAL_RADIUS: float = 1.0
GOAL_REWARD: float = 250.0
COLLISION_PENALTY: float = -100.0
STEP_PENALTY: float = -0.002

PROGRESS_WEIGHT: float = 4.0            # вес прямой дистанции до цели
PATH_PROGRESS_WEIGHT: float = 10.0      # вес приближения к следующей точке маршрута
PATH_FOLLOW_WEIGHT: float = 1.0         # штраф за отклонение от маршрута
SMOOTHNESS_PENALTY: float = 0.004
FORWARD_BONUS_WEIGHT: float = 0.08
STANDSTILL_PENALTY: float = 0.04


def wrap_angle(angle: chex.Array) -> chex.Array:
    """Нормализует угол в диапазон [-pi, pi]."""
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def world_to_vehicle(vec: chex.Array, heading: chex.Array) -> chex.Array:
    """Переводит вектор из мировых координат в систему координат машины."""
    c = jnp.cos(heading)
    s = jnp.sin(heading)
    return jnp.array([
        c * vec[0] + s * vec[1],
        -s * vec[0] + c * vec[1],
    ], dtype=jnp.float32)


class RobotaxiState(equinox.Module):
    time: chex.Scalar
    goal_pos: chex.Array
    agent_pos: chex.Array
    agent_forward_dir: chex.Array
    heading: chex.Scalar
    speed: chex.Scalar
    steering: chex.Scalar
    static_obstacles: chex.Array
    kinematic_obstacles: chex.Array
    kinematic_obst_velocities: chex.Array
    path_array: chex.Array


class RobotaxiObservation(equinox.Module):
    distance_to_path: chex.Array
    direction_of_path: chex.Array
    collision_rays: chex.Array
    speed: chex.Array
    steering: chex.Array
    goal_vec_local: chex.Array
    min_clearance: chex.Array
    turn_sin_path: chex.Array
    turn_cos_path: chex.Array
    turn_sin_goal: chex.Array
    turn_cos_goal: chex.Array

    def as_vector(self) -> chex.Array:
        return jnp.concatenate(
            [
                jnp.atleast_1d(self.distance_to_path),
                jnp.reshape(self.direction_of_path, (-1,)),
                jnp.reshape(self.collision_rays, (-1,)),
                jnp.atleast_1d(self.speed),
                jnp.atleast_1d(self.steering),
                jnp.reshape(self.goal_vec_local, (-1,)),
                jnp.atleast_1d(self.min_clearance),
                jnp.atleast_1d(self.turn_sin_path),
                jnp.atleast_1d(self.turn_cos_path),
                jnp.atleast_1d(self.turn_sin_goal),
                jnp.atleast_1d(self.turn_cos_goal),
            ],
            axis=-1,
        )


class RobotaxiEnv(BaseEnv):

    @staticmethod
    def _check_safe_collision(
        agent_pos: chex.Array,
        obstacles: chex.Array,
        step_time: chex.Scalar,
    ) -> chex.Array:
        if obstacles.size == 0:
            return jnp.array(False)

        dists = jnp.linalg.norm(obstacles - agent_pos[None, :], axis=-1)
        collision_threshold = OBSTACLE_RADIUS * ROBOT_COLLISION_MARGIN
        return jnp.any(dists < collision_threshold)

    @staticmethod
    def _compute_min_clearance(agent_pos: chex.Array, obstacles: chex.Array) -> chex.Array:
        if obstacles.size == 0:
            return jnp.asarray(10.0, dtype=jnp.float32)
        dists = jnp.linalg.norm(obstacles - agent_pos[None, :], axis=-1)
        clearance = jnp.min(dists) - (OBSTACLE_RADIUS + ROBOT_RADIUS)
        return clearance.astype(jnp.float32)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def _plan_safe_path(
        agent_pos: chex.Array,
        goal_pos: chex.Array,
        obstacles: chex.Array,
        env_params: BaseEnvParams,
    ) -> chex.Array:
        """
        Строит маршрут с учётом радиуса препятствия (1.0) и габаритов робота.
        Раздувает препятствия маской радиуса GRID_INFLATE_RADIUS.
        """
        h_max, w_max = env_params.map_height_width
        shape, dtype = env_params.nav_grid_shape_dtype

        to_map_vmap = jax.vmap(
            partial(utils.convert_to_map_view, map_shape=env_params.map_height_width)
        )

        obs_grid = BaseEnv._to_grid_pos(obstacles, env_params.discretization_scale)
        obs_map = to_map_vmap(obs_grid)
        obs_y = jnp.clip(obs_map[:, 0], 0, h_max - 1)
        obs_x = jnp.clip(obs_map[:, 1], 0, w_max - 1)

        start_map = utils.convert_to_map_view(agent_pos, env_params.map_height_width)
        start_grid = BaseEnv._to_grid_pos(start_map, env_params.discretization_scale)
        sy = jnp.clip(start_grid[0], 0, h_max - 1)
        sx = jnp.clip(start_grid[1], 0, w_max - 1)

        goal_map = utils.convert_to_map_view(goal_pos, env_params.map_height_width)
        goal_grid = BaseEnv._to_grid_pos(goal_map, env_params.discretization_scale)
        gy = jnp.clip(goal_grid[0], 0, h_max - 1)
        gx = jnp.clip(goal_grid[1], 0, w_max - 1)

        kernel_radius: int = 1
        side = 2 * kernel_radius + 1
        coords = jnp.arange(side, dtype=jnp.float32) - float(kernel_radius)
        yy, xx = jnp.meshgrid(coords, coords, indexing="ij")
        circle_mask = ((xx**2 + yy**2) <= (GRID_INFLATE_RADIUS**2)).astype(jnp.int16)

        raw_obs = jnp.zeros(shape, dtype=jnp.int16).at[obs_y, obs_x].set(1)
        inflated_obs = jax.lax.conv_general_dilated(
            lhs=raw_obs[None, None, :, :],
            rhs=circle_mask[None, None, :, :],
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0, 0]

        mask = jnp.where(inflated_obs > 0, True, False)
        mask = mask.at[sy, sx].set(False)
        mask = mask.at[gy, gx].set(False)

        V = jnp.full(shape, jnp.inf, dtype=dtype).at[gy, gx].set(0.0)
        N = max(env_params.path_array_shape_dtype[0])

        static_shifts = [
            (0, 1, 1.0),
            (0, -1, 1.0),
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (1, 1, 1.41421356),
            (-1, -1, 1.41421356),
            (-1, 1, 1.41421356),
            (1, -1, 1.41421356),
        ]
        obs_cost_val = jnp.asarray(OBSTACLE_COST, dtype=dtype)
        penalty_val = jnp.asarray(OBSTACLE_COST / 10.0, dtype=dtype)

        def dp_step(_, v_curr):
            cand_list = []
            for dy, dx, base_cost in static_shifts:
                shifted_v = jnp.roll(jnp.roll(v_curr, shift=dy, axis=0), shift=dx, axis=1)
                shifted_m = jnp.roll(jnp.roll(mask, shift=dy, axis=0), shift=dx, axis=1)
                step_c = jnp.where(shifted_m, obs_cost_val, jnp.asarray(base_cost, dtype=dtype))
                cand_list.append(shifted_v + step_c)

            best_neighbor = jnp.min(jnp.stack(cand_list, axis=0), axis=0)
            v_next = jnp.minimum(v_curr, best_neighbor)
            v_next = jnp.where(mask, v_next + penalty_val, v_next)
            return v_next

        V = jax.lax.fori_loop(0, N * 2, dp_step, init_val=V)

        trace_moves = jnp.array(
            [
                [0, 1],
                [0, -1],
                [-1, 0],
                [1, 0],
                [1, 1],
                [-1, -1],
                [-1, 1],
                [1, -1],
                [0, 0],
            ],
            dtype=jnp.int16,
        )

        def trace_step(pos, _):
            def eval_move(d):
                nxt = jnp.clip(
                    pos + d,
                    jnp.array([0, 0], dtype=jnp.int16),
                    jnp.array([h_max - 1, w_max - 1], dtype=jnp.int16),
                )
                return nxt, V.at[nxt[0], nxt[1]].get()

            cand_poses, cand_vals = jax.vmap(eval_move)(trace_moves)
            best_idx = jnp.argmin(cand_vals)
            nxt_pos = cand_poses[best_idx]
            return nxt_pos, nxt_pos

        _, path_grid = jax.lax.scan(
            trace_step, jnp.array([sy, sx], dtype=jnp.int16), length=N
        )
        path_world = jax.vmap(utils.convert_to_world_view, in_axes=(0, None))(
            path_grid, env_params.map_height_width
        )
        return path_world.astype(env_params.path_array_shape_dtype[1])

    @staticmethod
    def init_params(
        key: chex.PRNGKey,
        map_id: int = 0,
        max_steps: int = 1200,
        path_length: int = 100,
        discretization_scale: int = 1,
        perception_radius: float = 5.0,
        num_ray_sensors: int = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: int = FPS,
    ) -> Tuple[BaseEnvParams, RobotaxiState]:
        params, base_state = BaseEnv.init_params(
            key=key,
            map_id=map_id,
            max_steps=max_steps,
            path_length=path_length,
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            num_ray_sensors=num_ray_sensors,
            fov=fov,
            fps=fps,
        )

        obstacles = jnp.concatenate(
            [base_state.static_obstacles, base_state.kinematic_obstacles], axis=0
        )
        safe_path = RobotaxiEnv._plan_safe_path(
            base_state.agent_pos, base_state.goal_pos, obstacles, params
        )

        to_wp2 = safe_path[2] - base_state.agent_pos
        to_wp1 = safe_path[1] - base_state.agent_pos
        to_goal = base_state.goal_pos - base_state.agent_pos
        init_vec = jnp.where(
            jnp.linalg.norm(to_wp2) > 1e-4,
            to_wp2,
            jnp.where(jnp.linalg.norm(to_wp1) > 1e-4, to_wp1, to_goal),
        )
        init_heading = jnp.arctan2(init_vec[1], init_vec[0])
        init_dir = jnp.array(
            [jnp.cos(init_heading), jnp.sin(init_heading)], dtype=jnp.float32
        )

        state = RobotaxiState(
            time=jnp.asarray(0),
            goal_pos=base_state.goal_pos,
            agent_pos=base_state.agent_pos,
            agent_forward_dir=init_dir,
            heading=init_heading,
            speed=jnp.asarray(0.0, dtype=jnp.float32),
            steering=jnp.asarray(0.0, dtype=jnp.float32),
            static_obstacles=base_state.static_obstacles,
            kinematic_obstacles=base_state.kinematic_obstacles,
            kinematic_obst_velocities=base_state.kinematic_obst_velocities,
            path_array=safe_path,
        )
        return params, state

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(
        key: chex.PRNGKey,
        env_params: BaseEnvParams,
        init_state: RobotaxiState | None,
    ) -> Tuple[RobotaxiObservation, RobotaxiState]:
        if init_state is None:
            raise ValueError("init_state must not be None in RobotaxiEnv.reset")
        obs = RobotaxiEnv.get_observation(init_state, env_params)
        return obs, init_state

    @staticmethod
    def get_observation(
        env_state: RobotaxiState, env_params: BaseEnvParams
    ) -> RobotaxiObservation:
        base_obs = BaseEnv.get_observation(env_state, env_params)
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )

        rel_goal = env_state.goal_pos - env_state.agent_pos
        diag = jnp.sqrt(
            jnp.asarray(env_params.map_height_width[0], dtype=jnp.float32) ** 2
            + jnp.asarray(env_params.map_height_width[1], dtype=jnp.float32) ** 2
        )
        goal_local = world_to_vehicle(rel_goal, env_state.heading) / jnp.maximum(diag, 1e-6)
        path_dir_local = world_to_vehicle(base_obs.direction_of_path, env_state.heading)
        min_clear = RobotaxiEnv._compute_min_clearance(env_state.agent_pos, obstacles)

        dists_wp = jnp.linalg.norm(env_state.path_array - env_state.agent_pos[None, :], axis=-1)
        look_idx = jnp.minimum(jnp.argmin(dists_wp) + 3, env_state.path_array.shape[0] - 1)
        to_wp = env_state.path_array[look_idx] - env_state.agent_pos
        wp_angle = jnp.arctan2(to_wp[1], to_wp[0])
        err_path = wrap_angle(wp_angle - env_state.heading)

        goal_angle = jnp.arctan2(rel_goal[1], rel_goal[0])
        err_goal = wrap_angle(goal_angle - env_state.heading)

        return RobotaxiObservation(
            distance_to_path=base_obs.distance_to_path,
            direction_of_path=path_dir_local,
            collision_rays=base_obs.collision_rays,
            speed=env_state.speed / MAX_SPEED,
            steering=env_state.steering / MAX_STEER_ANGLE,
            goal_vec_local=goal_local,
            min_clearance=min_clear,
            turn_sin_path=jnp.sin(err_path),
            turn_cos_path=jnp.cos(err_path),
            turn_sin_goal=jnp.sin(err_goal),
            turn_cos_goal=jnp.cos(err_goal),
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: RobotaxiState,
        action: chex.Array,
        env_params: BaseEnvParams,
    ) -> Tuple[RobotaxiObservation, RobotaxiState, chex.Scalar, chex.Array, Dict[str, Any]]:
        dt = jnp.asarray(env_params.step_size, dtype=jnp.float32)
        act = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)

        throttle_brake = act[0]
        steer_cmd = act[1]

        accel_input = jnp.where(
            throttle_brake >= 0.0,
            throttle_brake * MAX_ACCEL,
            throttle_brake * MAX_BRAKE,
        )
        rolling = ROLLING_RESISTANCE * jnp.tanh(env_state.speed / 0.2)
        drag = LINEAR_DRAG * env_state.speed
        net_accel = accel_input - rolling - drag

        new_speed = jnp.clip(env_state.speed + net_accel * dt, MIN_SPEED, MAX_SPEED)
        target_steer = steer_cmd * MAX_STEER_ANGLE
        new_steering = env_state.steering + jnp.clip(
            target_steer - env_state.steering, -2.5 * dt, 2.5 * dt
        )

        yaw_rate = (new_speed / WHEELBASE) * jnp.tan(new_steering)
        new_heading = wrap_angle(env_state.heading + yaw_rate * dt)
        new_forward = jnp.array(
            [jnp.cos(new_heading), jnp.sin(new_heading)], dtype=jnp.float32
        )
        new_agent_pos = env_state.agent_pos + new_forward * (new_speed * dt)

        new_kin_obs = BaseEnv._move_kinematic_obstacles(env_state, env_params)
        obstacles = jnp.concatenate([env_state.static_obstacles, new_kin_obs], axis=0)

        new_time = env_state.time + 1
        collision_done = RobotaxiEnv._check_safe_collision(
            new_agent_pos, obstacles, new_time
        )
        goal_done = jnp.linalg.norm(new_agent_pos - env_state.goal_pos) < GOAL_RADIUS
        time_done = new_time >= env_params.max_steps_in_episode
        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        should_replan = jnp.equal(jnp.mod(new_time, REPLAN_EVERY_N_STEPS), 0)
        new_path = jax.lax.cond(
            should_replan,
            lambda _: RobotaxiEnv._plan_safe_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            operand=None,
        )

        old_dist = jnp.linalg.norm(env_state.goal_pos - env_state.agent_pos)
        new_dist = jnp.linalg.norm(new_agent_pos - env_state.goal_pos)
        goal_progress = old_dist - new_dist

        dists_old_path = jnp.linalg.norm(
            env_state.path_array - env_state.agent_pos[None, :], axis=-1
        )
        wp_idx = jnp.minimum(
            jnp.argmin(dists_old_path) + 0, env_state.path_array.shape[0] - 1
        )
        target_wp = env_state.path_array[wp_idx]

        old_wp_dist = jnp.linalg.norm(target_wp - env_state.agent_pos)
        new_wp_dist = jnp.linalg.norm(target_wp - new_agent_pos)
        path_progress = old_wp_dist - new_wp_dist

        dists_to_path = jnp.linalg.norm(new_path - new_agent_pos[None, :], axis=-1)
        min_path_dist = jnp.min(dists_to_path)

        forward_bonus = FORWARD_BONUS_WEIGHT * jnp.clip(new_speed / MAX_SPEED, 0.0, 1.0)
        standstill_pen = jnp.where(new_speed < 0.2, STANDSTILL_PENALTY, 0.0)

        reward = (
            PATH_PROGRESS_WEIGHT * path_progress
            + PROGRESS_WEIGHT * goal_progress
            + STEP_PENALTY
            - PATH_FOLLOW_WEIGHT * jnp.clip(min_path_dist, 0.0, 5.0)
            - SMOOTHNESS_PENALTY * jnp.sum(jnp.square(act))
            - standstill_pen
            + forward_bonus
            + jnp.where(goal_done, GOAL_REWARD, 0.0)
            + jnp.where(
                jnp.logical_and(collision_done, jnp.logical_not(goal_done)),
                COLLISION_PENALTY,
                0.0,
            )
        )

        next_state = RobotaxiState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_pos=new_agent_pos,
            agent_forward_dir=new_forward,
            heading=new_heading,
            speed=new_speed,
            steering=new_steering,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=new_kin_obs,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=new_path,
        )

        obs = RobotaxiEnv.get_observation(next_state, env_params)
        info = {
            "time": new_time,
            "speed": new_speed,
            "distance_to_goal": new_dist,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
            "collision": collision_done,
            "success": goal_done,
        }
        return obs, next_state, reward, done, info