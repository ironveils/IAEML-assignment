import jax

from env.robotaxi import RobotaxiEnv
from utils.renderer import PygameFrontend
from utils.autoreset import AutoResetWrapper


key = jax.random.PRNGKey(0)
env_params, init_state = RobotaxiEnv.init_params(
    key=key,
    map_id=0,
    max_steps=1200,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

env = AutoResetWrapper(RobotaxiEnv, env_params, init_state)
frontend = PygameFrontend(env, env_params, init_state, eval_mode=False)
frontend.run()
