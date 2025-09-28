#!/usr/bin/env python3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
from rl_drone_new import WildfireDroneEnv

def run_demo(
    model_path="./models/pyroguard_ppo_20250927_225702/final_model_ppo.zip",
    vec_path="./models/pyroguard_ppo_20250927_225702/vec_normalize.pkl",
    episodes=3,
    grid_size=25,
    render=True,
):
    print(f"📂 Loading trained model from {model_path}")

    # Load evaluation environment
    eval_env = WildfireDroneEnv(
        grid_size=grid_size,
        max_steps=800,
        enable_multi_agent=False,
        num_drones=1,
    )
    vec = DummyVecEnv([lambda: eval_env])

    if vec_path and os.path.exists(vec_path):
        vec = VecNormalize.load(vec_path, vec)
        vec.training = False
        vec.norm_reward = False

    # Load PPO model
    model = PPO.load(model_path)

    for ep in range(1, episodes + 1):
        obs = vec.reset()
        done = [False]
        total_reward = 0
        steps = 0
        print(f"\n🎮 Episode {ep}/{episodes}")

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec.step(action)
            total_reward += reward[0]
            steps += 1

            if render:
                try:
                    eval_env.render()
                except NotImplementedError:
                    # fallback if wrapper blocks render
                    vec.envs[0].render()

        print(f"✅ Episode finished: total reward={total_reward:.2f}, steps={steps}")

if __name__ == "__main__":
    run_demo()
