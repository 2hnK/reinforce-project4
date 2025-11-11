import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm

# from_checkpoint 는 체크포인트에 저장된 config를 그대로 로드함
ckpt_path = ""
algo_eval = Algorithm.from_checkpoint(ckpt_path) # 20227128: 알고리즘 로딩 활성화

# 에피소드 평가
def compute_action(obs):
    # 20227128: explore = 평가 시 탐험을 하지 않고 결정론적(deterministic)으로 행동
    action = algo_eval.compute_single_action(obs, explore=False)
    return action
    # return env.action_space.sample()

env = gym.make("HalfCheetah-v5")

NUM_EVAL_EPISODES = 10
returns = []

for ep in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        action = compute_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_ret += float(reward)
    returns.append(ep_ret)
    print(f"[EVAL] Episode {ep+1}/{NUM_EVAL_EPISODES}: return={ep_ret:.3f}")

mean_ret = float(np.mean(returns)) if returns else float("nan")
std_ret = float(np.std(returns)) if returns else float("nan")
print(f"[EVAL] Mean return over {NUM_EVAL_EPISODES} eps: {mean_ret:.3f} ± {std_ret:.3f}")

env.close()

algo_eval.stop() # 20227128: 리소스 정리
