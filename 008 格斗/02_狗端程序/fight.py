#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lite3 全自主格斗程序  FIGHT v1  (2026-11-20)

一句话：用狗自带的 USB 相机"看见"对手 → 转向对准 → 压低速度顶上去 → 丢目标时盲推一小段
→ 摔倒就停 → 再爬起来找。全程不需要人碰。

设计前提（现场规则未知，所以参数全部可调）
------------------------------------------
1) 只做「追 + 撞 + 顶」，不做跳跃/空翻：那些动作在 network.toml 里是开的，但触发码未验证，
   且在有人看着的课堂上跳跃撞狗有损坏风险。规则明确允许后再说。
2) 速度走 0x0140/0x0141 双精度通道（自主模式），不走摇杆轴通道：
   轴通道有 20% 死区（6555/32767），轻推根本发不出去；速度通道是真实 m/s。
   但两条通道都实现了，--chan axis 可一键切换（万一速度通道现场不走）。
3) 心率 4 Hz、速度帧 50 Hz、断心跳 1 s 狗自停；退出、Ctrl-C、超过 --duration 一律先零速再复位。
4) 默认 DRY-RUN：不加 --apply 只打印，不发一个字节。先在电脑上 / 在狗上趴着跑通再 --apply。

子命令
------
  python3 fight.py probe                 # 只读体检：相机/延迟/cv2/遥测/配置文件
  python3 fight.py calib --apply 0       # 相机调参：实时打印检测框，用来调阈值和量距离
  python3 fight.py run --apply           # 真打
  python3 fight.py replay --video x.mp4  # 用录好的视频离线跑一遍状态机（不碰狗）
  python3 fight.py selftest              # 纯逻辑自测，不需要相机和狗

常用组合（现场照着抄）
----------------------
  # 第一次、狗趴着、不上电：只体检
  python3 fight.py probe

  # 狗用架子架起（轮子空转也行），看感知是否正常、状态机是否在选对状态
  python3 fight.py calib --apply 0
  python3 fight.py run --apply --max-v 0.0     # 只走流程不动，看日志

  # 真正上场（推荐参数：中速步态、0.55 m/s 靠近、0.75 m/s 顶）
  python3 fight.py run --apply --preset push --gait medium

调试记录：--csv run.csv 写决策日志；--dump-dir frames 每 --dump-every 帧存一张标注图；
--video out.avi 存整段标注视频。比赛后回看这几样东西就能知道它为什么赢/输。
"""

import argparse
import math
import os
import signal
import socket
import struct
import sys
import threading
import time
import csv

VERSION = "FIGHT v1 · 2026-11-20"

# ============================ 协议常量（勿改） ============================
HB            = 0x21040001   # 心跳，>=4 Hz，缺 1 s 收回控制权
STAND_LIE     = 0x21010202   # 起立/趴下【切换】——注意是 toggle，连发两次等于没发
AUTO_MODE     = 0x21010C03   # 自主模式：只认 0x0140 速度指令
HANDLE_MODE   = 0x21010C02   # 手柄模式：认 0x21010130 轴指令
STAND_MODE    = 0x21010D05   # 原地模式（只有此模式能调姿态）
MOVE_MODE     = 0x21010D06   # 移动模式
CONT_GAIT     = 0x21010C06   # 连续运动步态，-1 开 / 2 关

GAIT_FAST     = 0x21010303   # 平地高速步态
GAIT_MEDIUM   = 0x21010307   # 平地中速步态

VEL_X         = 0x0140       # 前进速度 m/s（type=1 时后面跟 double）
VEL_Y         = 0x0145       # 横移速度 m/s
VEL_Z         = 0x0141       # 转向角速度 rad/s
VEL_VALUE_FLAG = 8
VEL_TYPE      = 1

AXIS_X        = 0x21010130   # 移动模式下：前后平移速度（摇杆通道）
AXIS_Y        = 0x21010131   # 移动模式下：左右平移速度
AXIS_Z        = 0x21010135   # 移动模式下：转向角速度
AXIS_MAX      = 32767
AXIS_DEAD     = 6553         # 死区：|raw| <= 6553 会被当 0

T0 = time.time()


def log(msg):
    print("[%7.2fs] %s" % (time.time() - T0, msg))
    sys.stdout.flush()


def warn(msg):
    print("[%7.2fs] !! %s" % (time.time() - T0, msg))
    sys.stdout.flush()


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def simple_frame(code, value=0, ctype=0):
    """12 字节：<3i  code, value, type"""
    return struct.pack("<3i", int(code), int(value), int(ctype))


def vel_frame(code, v):
    """20 字节：<3id  code, 8, 1, double  —— 官方"速度指令"帧"""
    return struct.pack("<3id", int(code), VEL_VALUE_FLAG, VEL_TYPE, float(v))


def axis_val(v, vmax):
    """m/s -> 摇杆轴值；死区内返回 0（发了也会被当 0，反而抖）。"""
    if abs(v) < 1e-6 or vmax <= 0:
        return 0
    n = clamp(v / float(vmax), -1.0, 1.0)
    raw = int(round(n * AXIS_MAX))
    if abs(raw) <= AXIS_DEAD:
        return 0
    return raw


# ============================ 链路层 ============================
class Link(object):
    """心跳 + 50 Hz 速度泵 + 加速度限幅。

    apply=False 时所有 sendto 都被吞掉（DRY-RUN），但状态机、日志、时序全部照常跑，
    可以在电脑上、狗趴着时把流程演一遍。
    sink：可选回调 data->None，自测用（替代 socket）。
    """

    def __init__(self, ip="127.0.0.1", port=43893, hz=50.0, hb_period=0.25,
                 chan="vel", axis_vmax=2.4, apply=True, sink=None):
        self.ip = ip
        self.port = int(port)
        self.dt = 1.0 / float(hz)
        self.hb_period = hb_period
        self.chan = chan
        self.axis_vmax = float(axis_vmax)
        self.apply = bool(apply)
        self.sink = sink
        self.sock = None
        if self.sink is None and self.apply:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)

        # 目标值 -> 实际发送值（限幅）
        self.t_vx = 0.0
        self.t_wz = 0.0
        self.vx = 0.0
        self.wz = 0.0
        self.a_max = 6.0          # m/s^2，0->0.75 约 0.12 s
        self.w_acc = 10.0         # rad/s^2
        self.armed = False
        self.running = False
        self.frames = 0
        self.hb_seq = 0
        self.last_send_ok = None
        self.send_errors = 0
        self._lock = threading.Lock()
        self._th = None

    # ---------- 生命周期 ----------
    def start(self):
        self.running = True
        self._th = threading.Thread(target=self._loop, name="link")
        self._th.daemon = True
        self._th.start()

    def stop(self):
        self.running = False
        self.armed = False
        if self._th:
            self._th.join(timeout=1.0)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

    # ---------- 发送 ----------
    def _send(self, data):
        if self.sink is not None:
            self.sink(data)
            self.frames += 1
            return
        if not self.apply:
            self.frames += 1          # DRY-RUN 也计数，便于看频率是否到 50 Hz
            return
        try:
            self.sock.sendto(data, (self.ip, self.port))
            self.frames += 1
        except OSError as e:
            self.send_errors += 1
            if self.send_errors <= 3:
                warn("UDP 发送失败：%s" % e)

    def send_simple(self, code, value=0):
        self._send(simple_frame(code, value))

    def set_vel(self, vx, wz):
        with self._lock:
            self.t_vx = float(vx)
            self.t_wz = float(wz)

    def zero_vel(self):
        with self._lock:
            self.t_vx = 0.0
            self.t_wz = 0.0
            self.vx = 0.0
            self.wz = 0.0

    def _ramp(self, dt):
        step_v = self.a_max * dt
        step_w = self.w_acc * dt
        with self._lock:
            tvx, twz = self.t_vx, self.t_wz
            if self.vx < tvx:
                self.vx = min(tvx, self.vx + step_v)
            else:
                self.vx = max(tvx, self.vx - step_v)
            if self.wz < twz:
                self.wz = min(twz, self.wz + step_w)
            else:
                self.wz = max(twz, self.wz - step_w)
            return self.vx, self.wz

    def _loop(self):
        next_t = time.perf_counter()
        last_hb = 0.0
        while self.running:
            now = time.perf_counter()
            if now - last_hb >= self.hb_period:
                self.hb_seq = (self.hb_seq % 0x7FFFFFFE) + 1
                self.send_simple(HB, self.hb_seq)
                last_hb = now
            if self.armed:
                vx, wz = self._ramp(self.dt)
                if self.chan == "axis":
                    self.send_simple(AXIS_X, axis_val(vx, self.axis_vmax))
                    self.send_simple(AXIS_Y, 0)
                    self.send_simple(AXIS_Z, axis_val(-wz, self.axis_vmax))
                else:
                    self._send(vel_frame(VEL_X, vx))
                    self._send(vel_frame(VEL_Y, 0.0))
                    self._send(vel_frame(VEL_Z, wz))
            next_t += self.dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()   # 落后了就重新对齐，不追帧

    # ---------- 模式序列 ----------
    def _reliable(self, code, value=0, times=3, gap=0.04):
        for _ in range(times):
            self.send_simple(code, value)
            time.sleep(gap)

    def startup(self, chan=None, stand=False, gait="medium", pose_wait=0.0):
        """上电到可发速度的完整序列。返回时 armed=True。"""
        chan = chan or self.chan
        if chan == "axis":
            log("1) 切手柄模式      0x%08X（轴指令只能走手柄通道）" % HANDLE_MODE)
            self._reliable(HANDLE_MODE)
        else:
            log("1) 切自主模式      0x%08X（速度指令 0x0140 只能走自主模式）" % AUTO_MODE)
            self._reliable(AUTO_MODE)

        if stand:
            log("2) 起立/趴下切换   0x%08X（toggle！只发一次）" % STAND_LIE)
            self.send_simple(STAND_LIE)
            time.sleep(2.0)

        log("3) 切移动模式      0x%08X" % MOVE_MODE)
        self._reliable(MOVE_MODE)
        log("4) 开连续运动步态  0x%08X value=-1" % CONT_GAIT)
        self._reliable(CONT_GAIT, -1)
        g = GAIT_FAST if gait == "fast" else GAIT_MEDIUM
        log("5) 步态            %s 0x%08X" % (gait, g))
        self._reliable(g, 0)
        if pose_wait > 0:
            time.sleep(pose_wait)
        with self._lock:
            self.vx = 0.0
            self.wz = 0.0
            self.t_vx = 0.0
            self.t_wz = 0.0
        self.armed = True
        log("6) 已 armed，开始发速度帧（50 Hz）")

    def shutdown(self, hold=0.5):
        """零速收敛 -> 关连续步态 -> 回手柄模式+中速档 -> 停心跳。"""
        self.armed = False
        with self._lock:
            self.t_vx = 0.0
            self.t_wz = 0.0
            self.vx = 0.0
            self.wz = 0.0
        # 直接发几帧零速，确保最后落在 0
        for _ in range(6):
            if self.chan == "axis":
                self.send_simple(AXIS_X, 0)
                self.send_simple(AXIS_Z, 0)
            else:
                self._send(vel_frame(VEL_X, 0.0))
                self._send(vel_frame(VEL_Z, 0.0))
            time.sleep(0.02)
        log("复位：关连续步态 0x%08X value=2" % CONT_GAIT)
        self._reliable(CONT_GAIT, 2, times=2)
        self._reliable(HANDLE_MODE, times=2)
        self._reliable(GAIT_MEDIUM, times=2)
        time.sleep(hold)
        self.stop()
        log("已停心跳（狗 1 s 后自行收回控制权）")


# ============================ 感知层 ============================
class FrameReader(object):
    """独立线程读相机，永远只留最新一帧（避免 V4L2 缓冲堆积造成 0.3 s+ 的滞后）。"""

    def __init__(self, source, width=640, height=480, fps=30, fourcc="MJPG",
                 v4l2=True):
        self.raw_source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.fourcc = fourcc
        self.v4l2 = v4l2
        self.cap = None
        self.latest = None
        self.seq = 0
        self.t = 0.0
        self.fails = 0
        self.running = False
        self._th = None
        self._lock = threading.Lock()

    def open(self):
        import cv2
        src = self.raw_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = None
        if isinstance(src, int) and self.v4l2 and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
        if cap is None:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("打不开视频源 %r（设备被占？/dev/video0 被 track 占着？）" % (self.raw_source,))
        if self.fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 关键：只要最新帧
        except Exception:
            pass
        self.cap = cap
        return self

    def actual(self):
        import cv2
        if not self.cap:
            return (0, 0, 0.0)
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(self.cap.get(cv2.CAP_PROP_FPS)))

    def start(self):
        self.running = True
        self._th = threading.Thread(target=self._loop, name="cam")
        self._th.daemon = True
        self._th.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.fails += 1
                time.sleep(0.005)
                continue
            with self._lock:
                self.latest = frame
                self.t = time.perf_counter()
                self.seq += 1

    def get(self):
        with self._lock:
            return self.seq, self.t, self.latest

    def close(self):
        self.running = False
        if self._th:
            self._th.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


class Detector(object):
    """运动检测：降采样 -> 背景建模(MOG2) -> 形态学 -> 最大连通块。

    为什么不用 YOLO：单目 + 干净场地 + 单一对手，运动块在 640x480 上只要十几毫秒；
    YOLOv5s RKNN 要 ~50 ms，还要赌狗上装了 Python 版 rknn 运行时。先跑快的，
    不好用再换（--detector yolo 留了接口位置，见 README 的计划 B）。
    """

    def __init__(self, work_w=320, min_area=200, var_thresh=24, history=90,
                 roi_top=0.0, roi_bottom=1.0, blur=5, dilate=9, warmup=0):
        import cv2
        self.cv2 = cv2
        self.work_w = int(work_w)
        self.min_area = int(min_area)
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=int(history), varThreshold=float(var_thresh), detectShadows=False)
        self.roi_top = float(roi_top)
        self.roi_bottom = float(roi_bottom)
        self.roi_dy = 0          # ROI 相对整帧的纵向偏移（只是为了画框对得上）
        self.k_blur = int(blur) | 1 if blur else 1
        self.k_dil = int(dilate) | 1 if dilate else 1
        self.warm_needed = int(warmup)
        self.warm_count = 0
        self.frames = 0
        self.last_warn = 0.0

    @property
    def warming(self):
        return self.warm_count < self.warm_needed

    def update(self, frame):
        """返回 dict(found, bearing, area, dist, bbox, cx, cy)。
        bearing: [-1,1]，-1=目标在画面最左；area: 目标框面积占整幅的比例。
        """
        cv2 = self.cv2
        h0, w0 = frame.shape[:2]
        if self.roi_top > 0.0 or self.roi_bottom < 1.0:
            y1 = int(h0 * self.roi_top)
            y2 = int(h0 * self.roi_bottom)
            frame = frame[y1:y2, :]
            self.roi_dy = y1
            h0 = y2 - y1
        else:
            self.roi_dy = 0
        scale = float(self.work_w) / float(w0)
        small = cv2.resize(frame, (self.work_w, max(1, int(h0 * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.k_blur > 1:
            gray = cv2.GaussianBlur(gray, (self.k_blur, self.k_blur), 0)
        m = self.mog.apply(gray)                       # 学习率用默认(自适应)
        _, m = cv2.threshold(m, 200, 255, cv2.THRESH_BINARY)
        if self.k_dil > 1:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self.k_dil, self.k_dil)))
            m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (self.k_dil, self.k_dil)))
        self.frames += 1
        self.warm_count += 1

        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        best = None
        best_area = 0.0
        for c in cnts:
            a = cv2.contourArea(c)
            if a > best_area:
                best_area = a
                best = c
        if best is None or best_area < self.min_area:
            return dict(found=False, bearing=0.0, area=0.0, dist=None,
                        bbox=None, cx=0, cy=0, mask_px=best_area)

        x, y, bw, bh = cv2.boundingRect(best)
        # 还原到原图坐标
        inv = 1.0 / scale if scale > 0 else 1.0
        bw0 = bw * inv
        bh0 = bh * inv
        cx0 = (x + bw / 2.0) * inv
        cy0 = (y + bh / 2.0) * inv
        bearing = clamp((cx0 - w0 / 2.0) / (w0 / 2.0), -1.0, 1.0)
        area = (bw0 * bh0) / float(w0 * h0)
        return dict(found=True, bearing=bearing, area=area, dist=None,
                    bbox=(int(x * inv), int(y * inv), int(bw0), int(bh0)),
                    cx=cx0, cy=cy0, mask_px=best_area)

    def annotate(self, frame, det, extra=""):
        """在帧上画框和准星，供 dump / video / --show 用。"""
        cv2 = self.cv2
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)
        if det and det.get("found"):
            x, y, bw, bh = det["bbox"]
            y = y + self.roi_dy
            col = (0, 220, 0) if abs(det["bearing"]) < 0.25 else (0, 180, 255)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
            cv2.putText(frame, "b=%+.2f a=%.3f" % (det["bearing"], det["area"]),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        if extra:
            cv2.putText(frame, extra, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1)
        return frame


class StateProbe(object):
    """被动监听 0x0901 状态帧（43897），只为摔倒/姿态判断；收不到也不影响别的。

    帧结构（手册 1.3.1 RobotStateUpload）：12 字节帧头 + int basic + int gait
    + double rpy[3]（roll, pitch, yaw）。手头没有文档里 roll/pitch 的确切偏移，
    只解析 yaw@36 与 basic@12 / gait@16，翻滚更可靠的是"双轴加速度"——没有就算。
    """

    def __init__(self, port=43897):
        self.port = int(port)
        self.sock = None
        self.count = 0
        self.latest = None
        self._running = False
        self._th = None

    def start(self):
        if self.port <= 0:
            return False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", self.port))
            s.settimeout(0.2)
            self.sock = s
        except OSError as e:
            warn("监听 %d 失败（被占用？）：%s" % (self.port, e))
            return False
        self._running = True
        self._th = threading.Thread(target=self._loop, name="state")
        self._th.daemon = True
        self._th.start()
        return True

    def _loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            self.count += 1
            if len(data) >= 44 and struct.unpack_from("<i", data, 0)[0] == 0x0901:
                try:
                    self.latest = dict(
                        t=time.perf_counter(),
                        basic=struct.unpack_from("<i", data, 12)[0],
                        gait=struct.unpack_from("<i", data, 16)[0],
                        yaw=struct.unpack_from("<d", data, 36)[0],
                    )
                except struct.error:
                    pass

    def stop(self):
        self._running = False
        if self._th:
            self._th.join(timeout=0.6)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass


# ============================ 决策层 ============================
PRESETS = {
    # 温和推挤：慢、稳、好控，最适合第一轮
    "push":   dict(max_v=0.55, push_v=0.75, kp_yaw=1.8, contact_area=0.11,
                   retreat_area=0.05, push_time=2.5, blind_time=1.5),
    # 冲撞：更快、更早接触，适合场地大/对手跑得快的组
    "ram":    dict(max_v=0.85, push_v=1.00, kp_yaw=1.4, contact_area=0.09,
                   retreat_area=0.04, push_time=2.0, blind_time=1.6),
    # 试探：极慢，用来第一天确认"它到底会不会顶"
    "gentle": dict(max_v=0.30, push_v=0.40, kp_yaw=2.2, contact_area=0.13,
                   retreat_area=0.06, push_time=3.0, blind_time=1.2),
}


class Brain(object):
    """IDLE -> SEEK -> AIM -> PUSH -> (BLIND) -> ... -> STOP/RECOVER

    每个状态只关心两件事：这一拍发多少 vx / wz，以及什么时候换状态。
    所有阈值都从 cfg 读，现场改参数不用改代码。
    """

    IDLE, SEEK, AIM, PUSH, RECOVER, STOP = "IDLE", "SEEK", "AIM", "PUSH", "RECOVER", "STOP"

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = self.IDLE
        self.t_state = 0.0
        self.t_last = 0.0
        self.last_b = 0.0
        self.last_area = 0.0
        self.lost_since = None
        self.scan_dir = 1.0
        self.scan_flip_at = 0.0
        self.engaged = False
        self.events = []          # (t, from, to, why)

    # ---------- 内部 ----------
    def _enter(self, state, t, why=""):
        if state != self.state:
            self.events.append((t, self.state, state, why))
            log("状态 %s -> %s  %s" % (self.state, state, why))
        self.state = state
        self.t_state = t

    def engage(self, t):
        if not self.engaged:
            self.engaged = True
            self._enter(self.SEEK, t, "开打")
            self.scan_flip_at = t + self.cfg.scan_flip

    def on_fall(self, t):
        self._enter(self.RECOVER, t, "检测到摔倒/翻转")
        self.lost_since = None

    def stop(self, t, why=""):
        self._enter(self.STOP, t, why)

    # ---------- 主 step ----------
    def step(self, t, det):
        c = self.cfg
        if self.t_last <= 0:
            self.t_last = t
        dt = max(1e-3, t - self.t_last)
        self.t_last = t

        found = bool(det and det.get("found"))
        if found:
            b = det["bearing"]
            if c.bearing_lpf < 1.0:
                self.last_b = self.last_b * c.bearing_lpf + b * (1.0 - c.bearing_lpf)
            else:
                self.last_b = b
            self.last_area = det["area"]
            self.lost_since = None
        else:
            if self.lost_since is None:
                self.lost_since = t

        if self.state == self.IDLE:
            return 0.0, 0.0
        if self.state == self.STOP:
            return 0.0, 0.0

        if self.state == self.RECOVER:
            if t - self.t_state > c.recover_time:
                self._enter(self.SEEK, t, "恢复完成")
            return 0.0, 0.0

        # ---- 丢目标的统一处理 ----
        lost_for = (t - self.lost_since) if self.lost_since else 0.0

        if self.state == self.SEEK:
            if found:
                self._enter(self.AIM, t, "看见目标")
                return self._aim(t, det, dt)
            if t >= self.scan_flip_at:
                self.scan_dir = -self.scan_dir
                self.scan_flip_at = t + c.scan_flip
            return 0.0, self.scan_dir * c.scan_w

        if self.state == self.AIM:
            if found:
                return self._aim(t, det, dt)
            if lost_for > c.lost_grace:
                self._enter(self.SEEK, t, "丢目标 %.2fs" % lost_for)
                return 0.0, self.scan_dir * c.scan_w
            return 0.0, 0.0

        if self.state == self.PUSH:
            if t - self.t_state > c.push_time:
                self._enter(self.AIM, t, "顶满 %.1fs 回收" % c.push_time)
                return self._aim(t, det, dt)
            if not found:
                if lost_for > c.blind_time:
                    self._enter(self.SEEK, t, "盲推 %.1fs 仍丢目标" % lost_for)
                    return 0.0, self.scan_dir * c.scan_w
                return c.push_v, 0.0          # 近距盲推：保持航向硬顶
            # 还有目标：保持顶，同时小修正
            wz = clamp(c.kp_yaw * 0.6 * self.last_b, -c.max_wz, c.max_wz)
            return c.push_v, wz

        return 0.0, 0.0

    def _aim(self, t, det, dt):
        c = self.cfg
        err = self.last_b
        wz = clamp(c.kp_yaw * err, -c.max_wz, c.max_wz)
        # 大角度先转，别一边转一边冲
        gate = max(0.12, 1.0 - abs(err) / c.aim_gate)
        vx = c.max_v * gate
        if abs(err) < c.contact_angle and _contact(det, c):
            self._enter(self.PUSH, t, "接触 area=%.3f" % det["area"])
            return c.push_v, wz * 0.5
        return vx, wz


def _contact(det, c):
    if not det or not det.get("found"):
        return False
    return det["area"] >= c.contact_area


# ============================ 记录 ============================
class Recorder(object):
    FIELDS = ["t", "state", "found", "bearing", "area", "vx", "wz", "note"]

    def __init__(self, csv_path=None):
        self.fh = None
        self.w = None
        self.path = csv_path
        if csv_path:
            self.fh = open(csv_path, "w", newline="", encoding="utf-8")
            self.w = csv.writer(self.fh)
            self.w.writerow(self.FIELDS)

    def row(self, t, state, det, vx, wz, note=""):
        if not self.w:
            return
        self.w.writerow(["%.3f" % t, state,
                         1 if (det and det.get("found")) else 0,
                         "%.3f" % (det["bearing"] if det and det.get("found") else 0.0),
                         "%.4f" % (det["area"] if det and det.get("found") else 0.0),
                         "%.3f" % vx, "%.3f" % wz, note])
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None
            self.w = None


# ============================ 配置 ============================
def build_cfg(args):
    """把命令行 + 预设合成一份扁平配置，Brain 只读它。"""
    p = dict(PRESETS["push"])
    p.update(PRESETS.get(args.preset, {}))
    c = argparse.Namespace()
    c.max_v = args.max_v if args.max_v is not None else p["max_v"]
    c.push_v = args.push_v if args.push_v is not None else p["push_v"]
    c.max_wz = float(args.max_wz)
    c.kp_yaw = float(args.kp_yaw) if args.kp_yaw is not None else p["kp_yaw"]
    c.contact_area = float(args.contact_area) if args.contact_area is not None else p["contact_area"]
    c.retreat_area = p["retreat_area"]
    c.push_time = float(args.push_time) if args.push_time is not None else p["push_time"]
    c.blind_time = float(args.blind_time) if args.blind_time is not None else p["blind_time"]
    c.aim_gate = float(args.aim_gate)
    c.contact_angle = float(args.contact_angle)
    c.bearing_lpf = float(args.bearing_lpf)
    c.scan_w = float(args.scan_w)
    c.scan_flip = float(args.scan_flip)
    c.lost_grace = float(args.lost_grace)
    c.recover_time = float(args.recover_time)
    return c


# ============================ 体检 / 调参 ============================
def cmd_probe(args):
    log("=========== 狗端体检（只读，不发任何运动指令） ===========")
    log("host      : %s" % socket.gethostname())
    log("python    : %s" % sys.version.replace("\n", " "))
    try:
        import cv2
        log("cv2       : %s" % cv2.__version__)
    except Exception as e:
        warn("cv2 导入失败：%s  —— 没有 OpenCV 这套程序跑不了（见 README 的「B 计划」）" % e)
        cv2 = None
    if cv2 is not None:
        try:
            import numpy
            log("numpy     : %s" % numpy.__version__)
        except Exception as e:
            warn("numpy: %s" % e)

    # 摄像头设备
    devs = []
    if os.path.isdir("/dev"):
        for name in sorted(os.listdir("/dev")):
            if name.startswith("video"):
                devs.append("/dev/" + name)
    log("video 设备: %s" % (", ".join(devs) if devs else "无（/dev/video* 不存在）"))

    # 帧率与延迟
    if cv2 is not None and devs:
        for dev in devs:
            try:
                r = FrameReader(dev, args.width, args.height, args.fps, args.fourcc).open()
                r.start()
            except Exception as e:
                warn("%s 打不开：%s" % (dev, e))
                continue
            w, h, f = r.actual()
            log("%s 打开成功：%dx%d @ %.1f fps（请求 %dx%d@%s, %s）"
                % (dev, w, h, f, args.width, args.height, args.fps, args.fourcc))
            t_read = []
            n = 0
            t_end = time.time() + args.probe_seconds
            seq_last = 0
            while time.time() < t_end and n < 300:
                t0 = time.perf_counter()
                seq, ts, fr = r.get()
                if seq != seq_last and fr is not None:
                    seq_last = seq
                    t_read.append((time.perf_counter() - ts) * 1000.0)
                time.sleep(0.001)
                n += 1
            r.close()
            if t_read:
                t_read.sort()
                log("  有效帧 %d，帧龄(采集->到手) 中位 %.1f ms / 90%% %.1f ms"
                    % (len(t_read), t_read[len(t_read) // 2], t_read[int(len(t_read) * 0.9)]))
            else:
                warn("  这段时间没取到帧！")
    elif cv2 is not None:
        warn("没有 /dev/video*，跳过相机测试")

    # UDP 发送通道（往自己回环发，绝不发给狗）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(simple_frame(HB, 1), ("127.0.0.1", 43999))
        s.close()
        log("udp       : 本机回环发送 OK")
    except Exception as e:
        warn("udp 回环失败：%s" % e)

    # 遥测
    sp = StateProbe(args.state_port)
    if sp.start():
        t_end = time.time() + 2.0
        while time.time() < t_end:
            time.sleep(0.05)
        log("遥测 %d 帧 %s" % (sp.count,
                              ("basic=%s gait=%s yaw=%.1f" % (sp.latest["basic"], sp.latest["gait"], sp.latest["yaw"]))
                              if sp.latest else "（本机不是上报目标，属正常）"))
        sp.stop()

    # 关键配置：现场实际生效的速度上限
    for p in ("/home/ysc/jy_exe/conf/flat_rl_config.yaml",):
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "max_velocity_target" in line:
                        log("生效速度上限 %s: %s" % (p, line.strip()))
        except Exception as e:
            warn("读不到 %s：%s" % (p, e))

    # jy_exe 在不在跑
    running = False
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open("/proc/%s/cmdline" % pid, "rb") as fh:
                    cmd = fh.read().replace(b"\x00", b" ").decode("utf-8", "replace")
                if "jy_exe" in cmd:
                    log("jy_exe     : 在跑 (pid %s) %s" % (pid, cmd.strip()[:80]))
                    running = True
                    break
            except Exception:
                continue
    except Exception:
        pass
    if not running:
        warn("jy_exe 没在跑？发什么指令狗都不会动")
    log("=========== 体检结束 ===========")
    return 0


def cmd_calib(args):
    """实时打印检测结果，用来调 min_area / var_thresh，并量出"1 m 时框多宽"。"""
    import cv2
    r = FrameReader(args.source, args.width, args.height, args.fps, args.fourcc).open().start()
    det = Detector(work_w=args.work_w, min_area=args.min_area,
                   var_thresh=args.var_thresh, roi_top=args.roi_top,
                   roi_bottom=args.roi_bottom, warmup=args.warmup_frames)
    log("调参模式：把对手/你自己在镜头前移动，看 bearing 和 area 的变化；Ctrl-C 退出")
    log("提示：1 m 处狗身宽约 0.35~0.45 m，用一个已知距离量一次框宽，就能算 --obj-w/--hfov")
    t_end = time.time() + args.seconds if args.seconds > 0 else 1e18
    last_print = 0.0
    try:
        while time.time() < t_end:
            seq, ts, frame = r.get()
            if frame is None:
                time.sleep(0.01)
                continue
            d = det.update(frame)
            if args.show or args.dump_dir or args.video:
                vis = det.annotate(frame.copy(), d, "CALIB")
                if args.dump_dir:
                    os.makedirs(args.dump_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(args.dump_dir, "calib_%06d.jpg" % det.frames), vis)
                if args.show:
                    cv2.imshow("calib", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            now = time.time()
            if now - last_print > 0.25:
                last_print = now
                if det.warming:
                    log("预热中 %d/%d" % (det.warm_count, det.warm_needed))
                elif d["found"]:
                    log("found  bearing=%+.3f  area=%.4f  bbox=%s  mask_px=%.0f"
                        % (d["bearing"], d["area"], str(d["bbox"]), d["mask_px"]))
                else:
                    log("none   mask_px=%.0f" % d["mask_px"])
    except KeyboardInterrupt:
        pass
    finally:
        r.close()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return 0


# ============================ 主循环 ============================
def cmd_run(args):
    import cv2
    cfg = build_cfg(args)
    log("=========== FIGHT %s ===========" % VERSION)
    log("通道 %s:%d  chan=%s  apply=%s" % (args.ip, args.port, args.chan, args.apply))
    log("预设 %s：max_v=%.2f push_v=%.2f kp_yaw=%.2f contact_area=%.2f"
        % (args.preset, cfg.max_v, cfg.push_v, cfg.kp_yaw, cfg.contact_area))
    if not args.apply:
        warn("DRY-RUN：不会向狗发送任何指令。确认无误后加 --apply")

    link = Link(args.ip, args.port, chan=args.chan, axis_vmax=args.axis_vmax,
                apply=args.apply)
    reader = FrameReader(args.source, args.width, args.height, args.fps,
                         args.fourcc).open().start()
    w, h, f = reader.actual()
    log("相机 %s：%dx%d @ %.1f fps" % (args.source, w, h, f))
    det = Detector(work_w=args.work_w, min_area=args.min_area,
                   var_thresh=args.var_thresh, roi_top=args.roi_top,
                   roi_bottom=args.roi_bottom, warmup=args.warmup_frames)
    brain = Brain(cfg)
    probe = StateProbe(args.state_port)
    if args.fall_imu:
        warn("--fall-imu 还没有实机验证过的字段（状态帧里 roll/pitch 的偏移未知），本次忽略；"
             "狗倒了会原地停住，不会自己爬起来")
    if args.state_port > 0 and probe.start():
        log("遥测监听已开（%d），仅用于记录，不参与控制" % args.state_port)
    rec = Recorder(args.csv)

    writer = None
    if args.video:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.video, fourcc, max(5.0, min(30.0, f or 20.0)), (w, h))
        log("录像 -> %s" % args.video)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    stop_flag = {"v": False}

    def on_sig(signum, frame):
        warn("收到信号 %s，准备停机" % signum)
        stop_flag["v"] = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    t_start = time.time()
    last_seq = -1
    last_frame_t = time.time()
    last_row = 0.0
    dump_i = 0
    reason = "时长到"
    try:
        # ---- 背景预热（不动机器人） ----
        if det.warm_needed > 0:
            log("背景预热 %d 帧（狗此时必须保持不动）" % det.warm_needed)
            while not stop_flag["v"] and det.warm_count < det.warm_needed:
                seq, ts, fr = reader.get()
                if fr is not None and seq != last_seq:
                    last_seq = seq
                    det.update(fr)
                time.sleep(0.01)
            log("预热完成")

        link.start()
        link.startup(chan=args.chan, stand=args.stand, gait=args.gait)
        if args.auto_start:
            if args.pure_timer:
                log("触发方式：armed 后固定 %.1fs 开打（不等看见目标）" % args.delay)
            else:
                log("触发方式：看见目标(面积>%.3f)立即开打，否则 %.1fs 后开打"
                    % (args.trigger_area, args.delay))
        else:
            log("触发方式：手动（按回车），无 tty 时直接开打")
        t_armed = time.time()
        t_end = t_armed + args.duration

        while not stop_flag["v"]:
            now = time.time()
            if now > t_end:
                reason = "时长到（%.0fs）" % args.duration
                break

            seq, ts, frame = reader.get()
            if frame is not None and seq != last_seq:
                last_seq = seq
                last_frame_t = now
                d = det.update(frame)
            else:
                d = None

            # 相机卡住 -> 立即零速（比丢目标更严重）
            if now - last_frame_t > 0.5:
                link.set_vel(0.0, 0.0)
                if d is None:
                    time.sleep(0.01)
                    continue
            elif d is None:
                # 没有新帧就沿用上一拍决策，速度泵照常发
                time.sleep(0.004)
                continue

            # 开打触发
            if not brain.engaged:
                if args.auto_start:
                    ready = (d.get("found") and d.get("area", 0) > args.trigger_area)
                    if (not args.pure_timer) and ready:
                        brain.engage(now)
                    elif now - t_armed > args.delay:
                        brain.engage(now)
                        if not (not args.pure_timer and ready):
                            log("延迟到点，开打")
                else:
                    if not args.no_pause and sys.stdin and sys.stdin.isatty():
                        input("按回车开打（--auto-start 可自动）")
                        brain.engage(now)
                    else:
                        brain.engage(now)

            vx, wz = brain.step(now, d)
            link.set_vel(vx, wz)

            # 记录 / 可视化
            if (writer is not None or args.dump_dir) and frame is not None:
                vis = det.annotate(frame.copy(), d,
                                   "%s vx=%.2f wz=%.2f" % (brain.state, vx, wz))
                if writer is not None:
                    writer.write(vis)
                if args.dump_dir and det.frames % max(1, args.dump_every) == 0:
                    cv2.imwrite(os.path.join(args.dump_dir, "f_%06d.jpg" % dump_i), vis)
                    dump_i += 1
            if args.show:
                cv2.imshow("fight", det.annotate(frame.copy(), d, "%s" % brain.state))
                if cv2.waitKey(1) & 0xFF == 27:
                    reason = "ESC"
                    break
            if now - last_row > 0.1:
                last_row = now
                rec.row(now - t_start, brain.state, d, vx, wz)

            # 50 Hz 主循环节拍（速度泵自己有线程，这里只是决策频率）
            time.sleep(max(0.0, 0.02 - (time.time() - now)))

    except Exception as e:
        reason = "异常：%s" % e
        import traceback
        traceback.print_exc()
    finally:
        try:
            link.zero_vel()
            link.shutdown()
        except Exception as e:
            warn("停机时出错：%s" % e)
        reader.close()
        recorder_rows = getattr(rec, "rows", 0)
        rec.close()
        if writer is not None:
            writer.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        probe.stop()
        log("结束原因：%s" % reason)
        log("共发帧 %d，状态迁移 %d 次，UDP 错误 %d"
            % (link.frames, len(brain.events), link.send_errors))
        for ev in brain.events[-12:]:
            log("  %6.2fs  %-7s -> %-7s %s" % (ev[0] - t_start, ev[1], ev[2], ev[3]))
    return 0


def cmd_replay(args):
    """用录好的视频离线跑状态机：可以反复调参，不碰狗、不需要相机。"""
    import cv2
    cfg = build_cfg(args)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        warn("打不开视频 %s" % args.video)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    det = Detector(work_w=args.work_w, min_area=args.min_area,
                   var_thresh=args.var_thresh, roi_top=args.roi_top,
                   roi_bottom=args.roi_bottom, warmup=args.warmup_frames)
    brain = Brain(cfg)
    rec = Recorder(args.csv)
    out = None
    if args.video_out:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    t = 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        t += 1.0 / fps
        d = det.update(frame)
        if det.warming:
            brain.step(t, None)
            continue
        if not brain.engaged:
            brain.engage(t)
        vx, wz = brain.step(t, d)
        rec.row(t, brain.state, d, vx, wz)
        if out is not None:
            out.write(det.annotate(frame.copy(), d, "%s vx=%.2f wz=%.2f" % (brain.state, vx, wz)))
    cap.release()
    rec.close()
    if out is not None:
        out.release()
    log("回放 %d 帧，状态迁移 %d 次" % (n, len(brain.events)))
    for ev in brain.events[:40]:
        log("  %6.2fs  %-7s -> %-7s %s" % (ev[0], ev[1], ev[2], ev[3]))
    return 0


# ============================ 自测 ============================
def cmd_selftest(args):
    """不需要相机、不需要狗。跑通说明协议字节和状态机逻辑是对的。"""
    log("=========== selftest ===========")
    # 1. 帧编码
    f = simple_frame(HB, 7)
    assert len(f) == 12 and struct.unpack("<3i", f) == (HB, 7, 0), "心跳帧错"
    f = vel_frame(VEL_X, 0.75)
    assert len(f) == 20 and struct.unpack("<3id", f) == (VEL_X, 8, 1, 0.75), "速度帧错"
    assert struct.unpack("<3i", f[:12]) == (VEL_X, 8, 1), "速度帧头错（value 必须是 8）"
    log("1) 字节协议 OK（心跳 12B / 速度 20B value=8 type=1）")

    # 2. 轴映射（含死区）
    assert axis_val(0.0, 2.4) == 0
    assert axis_val(0.05, 2.4) == 0                  # 0.05/2.4*32767=682 在死区内
    assert axis_val(1.2, 2.4) == 16384
    assert axis_val(2.4, 2.4) == AXIS_MAX
    assert axis_val(-9.0, 2.4) == -AXIS_MAX
    log("2) 轴映射 OK（死区 6553 / 满量程 32767）")

    # 3. Link：限幅 + 武装前后行为
    got = []
    link = Link(apply=True, sink=got.append, chan="vel")
    assert not link.armed
    link.armed = True
    link.set_vel(1.0, 0.5)
    link._ramp(0.02)
    assert 0 < link.vx < 1.0 and 0 < link.wz < 0.5
    for _ in range(50):
        link._ramp(0.02)
    assert abs(link.vx - 1.0) < 1e-6 and abs(link.wz - 0.5) < 1e-6
    link.send_simple(HB, 3)
    assert got and len(got[0]) == 12
    link.zero_vel()
    link._ramp(0.02)
    assert link.vx <= 1.0
    log("3) 链路限幅 OK（a_max=%.1f m/s^2，0->0.75 约 %.2fs）"
        % (6.0, 0.75 / 6.0))

    # 4. 状态机：看见 -> 靠近 -> 接触 -> 盲推 -> 搜索
    cfg = build_cfg(argparse.Namespace(
        preset="push", max_v=None, push_v=None, max_wz=1.2, kp_yaw=None,
        contact_area=None, push_time=None, blind_time=None, aim_gate=0.9,
        contact_angle=0.35, bearing_lpf=1.0, scan_w=0.7, scan_flip=1.2,
        lost_grace=0.6, recover_time=1.0))
    br = Brain(cfg)
    t = 0.0
    vx, wz = br.step(t, None)
    assert br.state == Brain.IDLE and vx == 0
    br.engage(t)
    assert br.state == Brain.SEEK
    # 目标在右侧 0.6
    vx, wz = br.step(t + 0.05, dict(found=True, bearing=0.6, area=0.01))
    assert br.state == Brain.AIM and wz > 0, "应该右转"
    assert vx > 0, "对准过程中也要前进"
    # 角度越大越慢
    vx_big, _ = br.step(t + 0.10, dict(found=True, bearing=0.6, area=0.01))
    vx_small, _ = br.step(t + 0.15, dict(found=True, bearing=0.05, area=0.01))
    assert vx_small > vx_big, "对准后才该加速"
    # 接触 -> PUSH
    vx, wz = br.step(t + 0.20, dict(found=True, bearing=0.02, area=0.30))
    assert br.state == Brain.PUSH and abs(vx - cfg.push_v) < 1e-9
    # 目标消失 -> 盲推
    t += 0.20
    vx, wz = br.step(t + 0.5, dict(found=False))
    assert br.state == Brain.PUSH and abs(vx - cfg.push_v) < 1e-9 and wz == 0
    # 盲推超时 -> SEEK
    vx, wz = br.step(t + cfg.push_time + 0.01, dict(found=False))
    assert br.state in (Brain.SEEK, Brain.AIM), br.state
    log("4) 状态机 OK（IDLE->SEEK->AIM->PUSH->盲推->回收）")

    # 5. 摔倒 -> RECOVER -> 恢复
    br.on_fall(10.0)
    assert br.state == Brain.RECOVER
    vx, wz = br.step(10.01, None)
    assert vx == 0 and wz == 0
    vx, wz = br.step(10.0 + cfg.recover_time + 0.01, None)
    assert br.state == Brain.SEEK
    log("5) 摔倒恢复 OK")

    # 6. 丢目标 -> 搜索，并且搜索方向会来回
    br.step(20.0, dict(found=True, bearing=0.0, area=0.02))
    br.step(20.1, dict(found=False))
    d1 = br.step(20.1 + cfg.lost_grace + 0.05, dict(found=False))[1]
    assert br.state == Brain.SEEK and d1 != 0, "丢目标应原地扫"
    log("6) 丢目标搜索 OK（wz=%+.2f rad/s）" % d1)

    log("=========== ALL OK · %d 项 ===========" % 6)
    return 0


# ============================ CLI ============================
def build_parser():
    p = argparse.ArgumentParser(
        description="Lite3 全自主格斗程序（默认 DRY-RUN，加 --apply 才动狗）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--source", default="0", help="0/1 = /dev/videoN；也可给 rtsp:// 或视频文件")
        sp.add_argument("--width", type=int, default=640)
        sp.add_argument("--height", type=int, default=480)
        sp.add_argument("--fps", type=float, default=30)
        sp.add_argument("--fourcc", default="MJPG")
        sp.add_argument("--work-w", type=int, default=320, help="检测降采样宽度")
        sp.add_argument("--min-area", type=int, default=200, help="最小运动块像素面积")
        sp.add_argument("--var-thresh", type=float, default=24, help="MOG2 灵敏度，越小越敏感")
        sp.add_argument("--roi-top", type=float, default=0.0)
        sp.add_argument("--roi-bottom", type=float, default=1.0)
        sp.add_argument("--warmup-frames", type=int, default=60, help="背景预热帧数")
        sp.add_argument("--show", action="store_true", help="弹窗显示（狗上没屏幕，别加）")
        sp.add_argument("--dump-dir", default=None, help="标注图存哪")
        sp.add_argument("--dump-every", type=int, default=15)
        sp.add_argument("--video", default=None, help="视频文件（run/calib = 录像输出；replay = 输入）")
        return sp

    def cfgargs(sp):
        sp.add_argument("--preset", default="push", choices=sorted(PRESETS.keys()))
        sp.add_argument("--max-v", type=float, default=None, help="靠近速度 m/s")
        sp.add_argument("--push-v", type=float, default=None, help="顶的速度 m/s")
        sp.add_argument("--max-wz", type=float, default=1.2, help="最大转向角速度 rad/s")
        sp.add_argument("--kp-yaw", type=float, default=None, help="转向比例系数")
        sp.add_argument("--contact-area", type=float, default=None, help="判定接触的目标面积占比")
        sp.add_argument("--push-time", type=float, default=None, help="一次顶多久后重新看一眼")
        sp.add_argument("--blind-time", type=float, default=None, help="丢了目标还能盲推多久")
        sp.add_argument("--aim-gate", type=float, default=0.9, help="偏多少角度就完全停下转向")
        sp.add_argument("--contact-angle", type=float, default=0.35, help="角度小于此值才算对准")
        sp.add_argument("--bearing-lpf", type=float, default=0.6, help="方位平滑 0=不平滑")
        sp.add_argument("--scan-w", type=float, default=0.7)
        sp.add_argument("--scan-flip", type=float, default=1.2)
        sp.add_argument("--lost-grace", type=float, default=0.6)
        sp.add_argument("--recover-time", type=float, default=1.0)

    sp = sub.add_parser("probe", help="只读体检")
    common(sp)
    sp.add_argument("--state-port", type=int, default=43897)
    sp.add_argument("--probe-seconds", type=float, default=3.0)

    sp = sub.add_parser("calib", help="相机调参")
    common(sp)
    sp.add_argument("--seconds", type=float, default=0)

    sp = sub.add_parser("run", help="真打（默认 DRY-RUN）")
    common(sp)
    cfgargs(sp)
    sp.add_argument("--apply", action="store_true", help="真的发指令给狗！")
    sp.add_argument("--ip", default="127.0.0.1", help="狗端 127.0.0.1；从 103 打就填 192.168.1.120")
    sp.add_argument("--port", type=int, default=43893)
    sp.add_argument("--chan", default="vel", choices=["vel", "axis"])
    sp.add_argument("--axis-vmax", type=float, default=2.4, help="axis 通道满量程，要等于 yaml 的 max_velocity_target[0]")
    sp.add_argument("--gait", default="medium", choices=["fast", "medium"])
    sp.add_argument("--stand", action="store_true", help="启动时发一次起立 toggle（狗已站着时别加）")
    sp.add_argument("--duration", type=float, default=180.0, help="最长运行时间，到点自动停")
    sp.add_argument("--auto-start", action="store_true", help="不等回车，按 --delay/看见目标自动开打")
    sp.add_argument("--pure-timer", action="store_true", help="只按时间开打，不等看见目标")
    sp.add_argument("--delay", type=float, default=5.0, help="armed 后等多久开打")
    sp.add_argument("--trigger-area", type=float, default=0.02, help="看见多大面积就提前开打")
    sp.add_argument("--no-pause", action="store_true", help="无 tty 时直接开打（脚本里用这个）")
    sp.add_argument("--state-port", type=int, default=0, help=">0 则监听遥测（默认关）")
    sp.add_argument("--fall-imu", action="store_true", help="用遥测判摔倒（未验证字段，谨慎）")
    sp.add_argument("--csv", default=None)

    sp = sub.add_parser("replay", help="回放视频跑状态机")
    common(sp)
    cfgargs(sp)
    sp.add_argument("--video-out", default=None, help="标注视频输出路径（可选）")
    sp.add_argument("--csv", default=None, help="决策日志输出（可选）")

    sp = sub.add_parser("selftest", help="纯逻辑自测")
    return p


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 2
    # probe/calib 没有 --csv；run 有
    if args.cmd == "probe":
        return cmd_probe(args)
    if args.cmd == "calib":
        return cmd_calib(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "replay":
        return cmd_replay(args)
    if args.cmd == "selftest":
        return cmd_selftest(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
