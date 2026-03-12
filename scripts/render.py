import torch
import hydra
import numpy as np
import einops
import time
import sys
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from isaaclab.app import AppLauncher

import wandb
import logging
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.helpers import make_env_policy, evaluate
import datetime
import termcolor

@hydra.main(config_path="../cfg", config_name="render", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    app_launcher = AppLauncher(OmegaConf.to_container(cfg.app))
    simulation_app = app_launcher.app

    # from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    # print("isaac dir:", ISAAC_NUCLEUS_DIR)
    # breakpoint()

    env, agent, vecnorm = make_env_policy(cfg)
    
    policy_eval = agent.get_rollout_policy("eval")
    evaluate(env, policy_eval, render=cfg.eval_render, render_mode=cfg.render_mode, seed=cfg.seed)
    os._exit(0)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
