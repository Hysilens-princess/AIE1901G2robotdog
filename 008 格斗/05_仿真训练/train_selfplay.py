"""回合制 self-play 训练 v2：混合对手池（fictitious self-play 简化版）。

关键发现（eval_baseline）：纯 self-play 快照对手会让策略过拟合「追逃跑者」——
vs 随机对手 0 胜 10 负 10 超时，vs 脚本逃跑者 20/20 全胜。
修法：每个 episode 从对手池按权重抽一种对手（policy 快照 / runner / defender / random），
让主策略面对多样行为，学会通用攻防而非只追逃跑者。

用法（PYTHONPATH=/126090458/deps:/opt/game2/deps）：
    python train_selfplay.py <out_dir> <total_steps> [--update-every 20000] [--device cpu]
"""
import os
import sys
import json
import time
import numpy as np
from fight_env import Lite3FightEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env


class OppHolder:
    def __init__(self, opp_model):
        self.opp = opp_model


class SelfPlayCallback(BaseCallback):
    def __init__(self, holder, save_path, update_every, verbose=0):
        super().__init__(verbose)
        self.holder = holder
        self.save_path = save_path
        self.update_every = update_every

    def _on_step(self):
        if self.n_calls % self.update_every == 0:
            self.model.save(self.save_path)
            self.holder.opp = PPO.load(self.save_path, device=self.model.device)
            if self.verbose:
                print(f"[selfplay] opponent updated at step {self.n_calls}")
        return True


def eval_fixed(model, n_episodes=20, seed=0):
    """vs 固定基线对手，各统计 ego 视角 胜/负/接触/超时。返回 dict[opp_mode] -> 统计。"""
    results = {}
    rng = np.random.default_rng(seed + 999)
    for opp_mode in ["random", "runner", "defender"]:
        env = Lite3FightEnv(role_mode="alternate", opp_mode=opp_mode)
        win = lose = contact = timeout = 0
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            ego = env._pose(env.j_ego)
            opp = env._pose(env.j_opp)
            if env._out_of_bounds(opp) and not env._out_of_bounds(ego):
                win += 1
            elif env._out_of_bounds(ego):
                lose += 1
            elif env._contact(ego, opp):
                contact += 1
            else:
                timeout += 1
        results[opp_mode] = {"win": win, "lose": lose, "contact": contact,
                             "timeout": timeout, "n": n_episodes}
    return results


def main():
    out = sys.argv[1]
    total = int(sys.argv[2])
    update_every = 20000
    device = "cpu"
    argv = sys.argv[3:]
    for i, a in enumerate(argv):
        if a == "--update-every":
            update_every = int(argv[i + 1])
        if a == "--device":
            device = argv[i + 1]

    os.makedirs(out, exist_ok=True)

    env = Lite3FightEnv(role_mode="alternate")
    model = PPO("MlpPolicy", env, verbose=1, device=device, n_steps=2048)

    holder = OppHolder(None)
    init_snap = os.path.join(out, "opp_init.zip")
    model.save(init_snap)
    holder.opp = PPO.load(init_snap, device=device)

    # 混合对手池：快照策略 40% + 三类脚本对手各 20%
    env.opp_policy = lambda obs: holder.opp.predict(obs, deterministic=False)[0]
    env.opp_pool = [(0.4, "policy"), (0.2, "runner"), (0.2, "defender"), (0.2, "random")]
    model.set_env(env)

    check_env(env, warn=True)

    snap = os.path.join(out, "opp_snapshot.zip")
    cb = SelfPlayCallback(holder, snap, update_every, verbose=1)

    t0 = time.time()
    model.learn(total_timesteps=total, callback=cb)
    dt = time.time() - t0
    model.save(os.path.join(out, "policy.zip"))

    metrics = eval_fixed(model, n_episodes=20)
    metrics["total_timesteps"] = total
    metrics["wall_seconds"] = round(dt, 1)
    metrics["device"] = device
    print("EVAL", json.dumps(metrics, ensure_ascii=False))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"saved {out}/policy.zip + metrics.json")


if __name__ == "__main__":
    main()
