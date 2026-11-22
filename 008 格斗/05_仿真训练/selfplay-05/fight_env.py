"""Lite3 格斗仿真环境（MuJoCo 版）— 回合制 self-play。

机身物理来自运动开发手册 V2.0.1（11kg / 548x370mm / 机身惯性真实）；
驱动用速度级 qvel 控制 + 加速度限幅，对齐真机「速度指令 → jy_exe ramp」；
传感器分两类：
  - 几何传感器真实模拟：超声波(mj_ray)、IMU(原生 sensor)、边界距离(解析)；
  - 检测器语义模拟：对手真值相对位姿 → FOV 裁剪 + 距离噪声 + 漏检 → (方位, 距离, 置信度)。
规则：5 米圆场地、禁止身体接触(接触双负)、逼对方出界=胜、出界方负；
  静止超 3 秒得 0 分（判负）、超时双方都得 0 分（2026-11-22 新规则）。

回合制：每 episode = 一个回合，role 由 reset 分配（交替或随机），
  +1=进攻方（施压逼对方出界）、-1=防守方（闪避存活）。
self-play：对手也用同一个策略（opp_policy 注入），拿相反 role 的观测，
  训练同一张策略网络同时学会进攻与防守。

用法（需 PYTHONPATH=/126090458/deps:/opt/game2/deps）：
    python fight_env.py smoke      # 自检
"""
import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

# ---------------- 常量（对齐真机 config / 手册） ----------------
VX_MAX, VY_MAX, WZ_MAX = 1.2, 1.0, 3.0        # max_velocity_target
A_MAX_LIN = 6.0                                # m/s^2 线加速度上限（ramp）
A_MAX_ANG = 15.0                               # rad/s^2 角加速度上限
ARENA_R = 2.5                                  # 5 米圆半径
DT = 0.02
BODY_HALF_LEN = 0.274                          # 机身半长（548mm/2）
BODY_HALF_WID = 0.185                          # 机身半宽（370mm/2）

# 超声波
US_MAX_RANGE = 4.0
US_NOISE = 0.02

# 检测器语义模拟（先猜参数，留标定接口，周五真机回填）
DET_FOV = math.radians(70)
DET_MISS_P = 0.05
DET_DIST_NOISE = 0.08
DET_BEARING_NOISE = 0.03
DET_CONF_GAIN = 0.7

CONTACT_DIST = 0.55

# 奖励
REWARD_WIN = 10.0       # 对方出界 / 对方静止判负
REWARD_LOSE = -10.0     # 自己出界 / 自己静止判负
REWARD_CONTACT = -10.0  # 接触双负
REWARD_STEP = -0.01     # 步进惩罚
REWARD_TIMEOUT = 0.0    # 超时双方 0 分（中性，新规则）
PROGRESS_GAIN = 1.0     # 进攻方「对手半径增量」施压奖励 / 防守方「自身半径减量」存活奖励
APPROACH_GAIN = 0.6     # 进攻方「逼近对手」奖励（缩小与对手距离）
SELF_KEEP_GAIN = 0.6    # 进攻方「别自己跑出界」惩罚（自身半径增大就扣）
EVADE_GAIN = 0.3        # 防守方「远离进攻方」奖励（拉开与对手距离）

# 静止判负（新规则：保持静止超 3 秒得 0 分）
STILL_SPEED = 0.03      # m/s，线速度低于此值视为「没在移动」
STILL_STEPS = 150       # 连续静止步数 = 3 秒（DT=0.02）
STILL_PRESSURE = 0.02   # 速度趋零的每步轻微惩罚（帮助信用分配，学「别停」）


class Lite3FightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opp_policy=None, role_mode="alternate", opp_mode="policy",
                 opp_pool=None, turn_steps=400, xml_path=None, progress_gain=PROGRESS_GAIN):
        super().__init__()
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "lite3_fight.xml")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        self.opp_policy = opp_policy      # callable(obs)->action；opp_mode="policy" 时用
        self.role_mode = role_mode        # "alternate" 或 "random"
        # opp_mode: "policy"(用 opp_policy) / "runner"(脚本逃跑者) /
        #           "defender"(脚本防守者，往圆心躲+远离进攻方) / "random"(随机动作)
        self.opp_mode = opp_mode
        # opp_pool: [(weight, mode_str), ...]，每个 episode 在 reset 时按权重抽一个对手模式
        #   用于让主策略面对「多样对手」避免过拟合单一逃跑行为
        self.opp_pool = opp_pool
        self._cur_opp_mode = opp_mode
        self.turn_steps = turn_steps
        self.progress_gain = progress_gain
        self._seed_rng = np.random.default_rng()

        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.j_ego = (jid("ego_x"), jid("ego_y"), jid("ego_yaw"))
        self.j_opp = (jid("opp_x"), jid("opp_y"), jid("opp_yaw"))
        self.b_ego = bid("ego")
        self.b_opp = bid("opp")

        # 观测 10 维（见 _observe）
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        self.role = 1.0
        self.step_count = 0
        self._episode = 0

    # ---------------- 状态存取 ----------------
    def _pose(self, j):
        return np.array([self.d.qpos[j[0]], self.d.qpos[j[1]], self.d.qpos[j[2]]])

    def _vel(self, j):
        return np.array([self.d.qvel[j[0]], self.d.qvel[j[1]], self.d.qvel[j[2]]])

    def _set_vel(self, j, target_vx, target_vy, target_wz):
        for idx, cur, tgt, amax in [
            (j[0], self.d.qvel[j[0]], target_vx, A_MAX_LIN),
            (j[1], self.d.qvel[j[1]], target_vy, A_MAX_LIN),
            (j[2], self.d.qvel[j[2]], target_wz, A_MAX_ANG),
        ]:
            err = tgt - cur
            self.d.qvel[idx] = cur + float(np.clip(err, -amax * DT, amax * DT))

    # ---------------- 传感器 ----------------
    @staticmethod
    def _body_frame(pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _ultrasound_front(self, pose, exclude_body):
        R = self._body_frame(pose)
        pnt = np.array([pose[0], pose[1], 0.33]) + R @ np.array([BODY_HALF_LEN + 0.01, 0, 0.02])
        vec = R @ np.array([1.0, 0, 0])
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(self.m, self.d, pnt, vec,
                             np.ones(6, dtype=np.uint8), 1, exclude_body, gid)
        if dist < 0:
            dist = US_MAX_RANGE
        dist = min(dist, US_MAX_RANGE)
        dist += float(self._seed_rng.normal(0.0, US_NOISE))
        return max(0.0, dist)

    def _detector(self, self_pose, other_pose):
        rel = other_pose[:2] - self_pose[:2]
        dist = float(np.linalg.norm(rel))
        if dist < 1e-6:
            return None
        R = self._body_frame(self_pose)
        local = R.T @ np.array([rel[0], rel[1], 0.0])
        bearing = math.atan2(local[1], local[0])
        if abs(bearing) > DET_FOV:
            return None
        if self._seed_rng.random() < DET_MISS_P:
            return None
        dist_noisy = dist + float(self._seed_rng.normal(0.0, DET_DIST_NOISE * (0.5 + dist / (2 * ARENA_R))))
        bearing_noisy = bearing + float(self._seed_rng.normal(0.0, DET_BEARING_NOISE))
        conf = float(np.clip(1.0 - DET_CONF_GAIN * dist / (2 * ARENA_R), 0.05, 1.0))
        conf = float(np.clip(conf + self._seed_rng.normal(0.0, 0.05), 0.0, 1.0))
        return (math.sin(bearing_noisy), math.cos(bearing_noisy),
                float(np.clip(dist_noisy / (2 * ARENA_R), 0.0, 1.0)), conf)

    def _observe(self, self_pose, self_vel, other_pose, role, self_body):
        """从 self 视角观测，role 是 self 的角色，self_body 是 self 的 body id。"""
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = np.clip(self_vel[0] / VX_MAX, -1, 1)
        obs[1] = np.clip(self_vel[1] / VY_MAX, -1, 1)
        obs[2] = np.clip(self_vel[2] / WZ_MAX, -1, 1)
        obs[3] = self._ultrasound_front(self_pose, self_body) / US_MAX_RANGE
        det = self._detector(self_pose, other_pose)
        if det is not None:
            obs[4], obs[5], obs[6], obs[7] = det
        else:
            obs[4], obs[5], obs[6], obs[7] = 0.0, 1.0, 1.0, 0.0
        r = float(np.linalg.norm(self_pose[:2]))
        obs[8] = float(np.clip((ARENA_R - r) / ARENA_R, -1, 1))
        obs[9] = float(role)
        return obs

    # ---------------- 规则判定 ----------------
    @staticmethod
    def _out_of_bounds(pose):
        return float(np.linalg.norm(pose[:2])) > ARENA_R

    @staticmethod
    def _contact(ego_pose, opp_pose):
        return float(np.linalg.norm(ego_pose[:2] - opp_pose[:2])) < CONTACT_DIST

    # ---------------- 对手动作：self-play 策略 或 脚本对手 ----------------
    def _opp_action(self, ego_pose, opp_pose, opp_vel):
        mode = self._cur_opp_mode
        if mode == "policy" and self.opp_policy is not None:
            opp_role = -self.role
            opp_obs = self._observe(opp_pose, opp_vel, ego_pose, opp_role, self.b_opp)
            a = np.asarray(self.opp_policy(opp_obs), dtype=np.float64)
            return float(a[0]) * VX_MAX, float(a[1]) * VY_MAX, float(a[2]) * WZ_MAX
        if mode == "random":
            a = self._seed_rng.uniform(-1.0, 1.0, 3)
            return float(a[0]) * VX_MAX, float(a[1]) * VY_MAX, float(a[2]) * WZ_MAX
        # runner 与 defender 都是「脚本逃跑者」家族：朝远离我方方向 + 切向 + 回圆心
        rel = opp_pose[:2] - ego_pose[:2]
        dist = float(np.linalg.norm(rel))
        away = rel / dist if dist > 1e-6 else np.array([1.0, 0.0])
        r = float(np.linalg.norm(opp_pose[:2]))
        if mode == "defender":
            # 移动躲避（新规则：不能静止超 3 秒）：绕场切向移动为主 + 贴边内收 + 被逼近加速躲避
            if r < 1e-3:
                centrip = np.array([0.0, 0.0])
                tang = np.array([0.0, 1.0])
            else:
                centrip = -opp_pose[:2] / r
                tang = np.array([-centrip[1], centrip[0]])
            edge_ratio = float(np.clip((r - 0.65 * ARENA_R) / (0.35 * ARENA_R), 0.0, 1.0))
            close_ratio = float(np.clip(1.0 - dist / (0.7 * ARENA_R), 0.0, 1.0))
            heading = (tang * (0.6 + 0.3 * close_ratio)
                       + centrip * (0.5 * edge_ratio)
                       + away * (0.15 * close_ratio))
        else:  # runner
            tang = np.array([-away[1], away[0]])
            centrip = -opp_pose[:2] / (r + 1e-6)
            heading = away * 0.7 + tang * 0.2 + centrip * (0.3 if r > ARENA_R - 0.5 else 0.0)
        heading /= np.linalg.norm(heading) + 1e-6
        yaw = opp_pose[2]
        target_yaw = math.atan2(heading[1], heading[0])
        err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        wz = float(np.clip(err * 3.0, -WZ_MAX, WZ_MAX))
        vx = 0.5 * float(math.cos(err)) if abs(err) < math.pi / 2 else 0.0
        return vx, 0.0, wz

    # ---------------- gym 接口 ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("start").id)
        mujoco.mj_forward(self.m, self.d)
        self.step_count = 0
        self._still_ego = 0
        self._still_opp = 0
        # 回合制角色分配
        if self.role_mode == "random":
            self.role = 1.0 if self._seed_rng.random() < 0.5 else -1.0
        elif self.role_mode == "attack":
            self.role = 1.0
        elif self.role_mode == "defend":
            self.role = -1.0
        else:  # alternate
            self.role = 1.0 if (self._episode % 2 == 0) else -1.0
        self._episode += 1
        # 每 episode 从 opp_pool 抽一个对手模式
        if self.opp_pool is not None:
            modes = [m for _, m in self.opp_pool]
            weights = [w for w, _ in self.opp_pool]
            self._cur_opp_mode = self._seed_rng.choice(modes, p=np.array(weights) / sum(weights))
        else:
            self._cur_opp_mode = self.opp_mode

        ego_pose = self._pose(self.j_ego)
        ego_vel = self._vel(self.j_ego)
        opp_pose = self._pose(self.j_opp)
        obs = self._observe(ego_pose, ego_vel, opp_pose, self.role, self.b_ego)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        vx = action[0] * VX_MAX
        vy = action[1] * VY_MAX
        wz = action[2] * WZ_MAX

        ego_pose = self._pose(self.j_ego)
        opp_pose = self._pose(self.j_opp)
        opp_vel = self._vel(self.j_opp)
        o_vx, o_vy, o_wz = self._opp_action(ego_pose, opp_pose, opp_vel)

        # 进度记录（施压/存活/逼近 shaping）
        opp_r_before = float(np.linalg.norm(opp_pose[:2]))
        ego_r_before = float(np.linalg.norm(ego_pose[:2]))
        dist_before = float(np.linalg.norm(opp_pose[:2] - ego_pose[:2]))

        self._set_vel(self.j_ego, vx, vy, wz)
        self._set_vel(self.j_opp, o_vx, o_vy, o_wz)
        mujoco.mj_step(self.m, self.d)

        ego_pose = self._pose(self.j_ego)
        ego_vel = self._vel(self.j_ego)
        opp_pose = self._pose(self.j_opp)
        opp_vel = self._vel(self.j_opp)
        self.step_count += 1

        # 静止检测（新规则：静止超 3 秒得 0 分判负）
        ego_speed = float(np.hypot(ego_vel[0], ego_vel[1]))
        opp_speed = float(np.hypot(opp_vel[0], opp_vel[1]))
        self._still_ego = self._still_ego + 1 if ego_speed < STILL_SPEED else 0
        self._still_opp = self._still_opp + 1 if opp_speed < STILL_SPEED else 0
        ego_still = self._still_ego >= STILL_STEPS
        opp_still = self._still_opp >= STILL_STEPS

        obs = self._observe(ego_pose, ego_vel, opp_pose, self.role, self.b_ego)

        ego_out = self._out_of_bounds(ego_pose)
        opp_out = self._out_of_bounds(opp_pose)
        contact = self._contact(ego_pose, opp_pose)
        timeout = self.step_count >= self.turn_steps

        # 奖励（从 ego 视角）
        reward = REWARD_STEP
        if ego_out:
            reward = REWARD_LOSE
        elif opp_out:
            reward = REWARD_WIN
        elif ego_still:
            reward = REWARD_LOSE             # 我方静止超 3 秒，判负得 0 分
        elif opp_still:
            reward = REWARD_WIN              # 对方静止超 3 秒，判负
        elif contact:
            reward = REWARD_CONTACT
        elif timeout:
            reward = REWARD_TIMEOUT          # 0.0：超时双方都得 0 分
        else:
            # shaping：
            #   进攻(role>0)：逼近对手 + 把对手往外逼 + 别自己跑出界
            #   防守(role<0)：留在场内 + 拉开与对手距离
            opp_r_after = float(np.linalg.norm(opp_pose[:2]))
            ego_r_after = float(np.linalg.norm(ego_pose[:2]))
            dist_after = float(np.linalg.norm(opp_pose[:2] - ego_pose[:2]))
            if self.role > 0:
                reward += self.progress_gain * (opp_r_after - opp_r_before)      # 施压：对手被往外逼
                reward += APPROACH_GAIN * (dist_before - dist_after)             # 逼近：拉近与对手
                reward -= SELF_KEEP_GAIN * max(0.0, ego_r_after - ego_r_before)  # 别自己跑出界
            else:
                reward += self.progress_gain * (ego_r_before - ego_r_after)      # 留场内：自身半径减小
                reward += EVADE_GAIN * (dist_after - dist_before)                # 拉开与对手距离
            # 静止压力：速度趋零给轻微负奖励，引导「保持移动」（帮助信用分配）
            reward -= STILL_PRESSURE * max(0.0, 1.0 - ego_speed / STILL_SPEED)

        terminated = ego_out or opp_out or contact or ego_still or opp_still or timeout
        return obs, reward, terminated, False, {}

    def close(self):
        pass


def smoke():
    env = Lite3FightEnv()
    obs, _ = env.reset()
    print(f"obs_space={env.observation_space.shape} action_space={env.action_space.shape}")
    print(f"role={env.role} obs={obs}")
    rng = np.random.default_rng(0)
    total_r = 0.0
    for i in range(400):
        a = rng.uniform(-1, 1, 3)
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term:
            eo = Lite3FightEnv._out_of_bounds(env._pose(env.j_ego))
            oo = Lite3FightEnv._out_of_bounds(env._pose(env.j_opp))
            ct = Lite3FightEnv._contact(env._pose(env.j_ego), env._pose(env.j_opp))
            print(f"step {i}: term, r={r:.2f} ego_out={eo} opp_out={oo} contact={ct} role={env.role}")
            break
    print(f"smoke OK: total_r={total_r:.2f} steps={env.step_count}")


if __name__ == "__main__":
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "smoke":
        smoke()
