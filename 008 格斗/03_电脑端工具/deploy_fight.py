# -*- coding: utf-8 -*-
"""
把格斗程序部署到狗的运动主机（192.168.1.120）并体检。

  python deploy_fight.py            上传 + 语法检查 + 狗端体检（不动狗）
  python deploy_fight.py --probe    再加一步相机/UDP/遥测体检（只读，仍不动狗）
  python deploy_fight.py --host 192.168.2.1     走狗的 WiFi 热点时用这个

依赖：pip install paramiko（本机已有）
密码：默认是单引号，可用环境变量 FIGHT_PWD 覆盖

注意：本脚本不发任何运动指令；上传后做的都是 ls / py_compile / bash check 这类只读动作。
"""

import argparse
import os
import socket
import sys
import time

try:
    import paramiko
except ImportError:
    sys.exit("缺 paramiko：pip install paramiko")

HOST_DEF = "192.168.1.120"
USER = "ysc"
PWD_DEF = chr(39)                      # 狗上通用密码就是一个单引号
REMOTE_DEF = "/home/ysc/fight"

HERE = os.path.dirname(os.path.abspath(__file__))          # 03_电脑端工具
PKG = os.path.dirname(HERE)                                # 008 格斗
ENG = os.path.join(PKG, "02_狗端程序")
FILES = ["fight.py", "run_fight.sh"]

KEY_CANDIDATES = [
    os.path.join(HERE, "ssh", "id_race"),
    os.path.join(os.path.expanduser("~"), ".ssh", "id_race"),
    os.path.join(os.path.expanduser("~"), ".ssh", "id_rsa"),
]


def port_open(host, port=22, timeout=3.0):
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError as e:
        print("[!] %s:%d 连不上：%s" % (host, port, e))
        return False
    finally:
        s.close()


def connect(host, pwd):
    for p in KEY_CANDIDATES:
        if not os.path.isfile(p):
            continue
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, username=USER, key_filename=p, timeout=8,
                      look_for_keys=False, allow_agent=False)
            print("[OK] 密钥登录：%s" % p)
            return c
        except Exception as e:
            print("[!] 密钥 %s 失败：%s" % (os.path.basename(p), e))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=USER, password=pwd, timeout=8,
              look_for_keys=False, allow_agent=False)
    print("[OK] 密码登录")
    return c


def run(c, cmd, timeout=60, quiet=False):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = (o.read() + e.read()).decode("utf-8", "replace")
    if not quiet:
        print("$ " + cmd)
        if out.strip():
            print(out.rstrip())
        print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST_DEF)
    ap.add_argument("--remote", default=REMOTE_DEF)
    ap.add_argument("--probe", action="store_true", help="额外跑相机/UDP/遥测体检")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()
    pwd = os.environ.get("FIGHT_PWD", PWD_DEF)

    missing = [f for f in FILES if not os.path.isfile(os.path.join(ENG, f))]
    if missing:
        sys.exit("本地缺文件：%s（在 %s）" % (", ".join(missing), ENG))

    print("本地程序目录：%s" % ENG)
    for f in FILES:
        p = os.path.join(ENG, f)
        print("  %-14s %6d 字节" % (f, os.path.getsize(p)))
    print()

    if not port_open(args.host):
        sys.exit("先把电脑网线接到狗的运动主机，或用狗的热点（--host 192.168.2.1）")

    c = connect(args.host, pwd)
    t0 = time.time()
    try:
        run(c, "mkdir -p %s && chmod 755 %s" % (args.remote, args.remote), quiet=True)
        sftp = c.open_sftp()
        for f in FILES:
            local = os.path.join(ENG, f)
            remote = "%s/%s" % (args.remote, f)
            sftp.put(local, remote)
            print("[OK] 上传 %s -> %s (%d 字节)" % (f, remote, os.path.getsize(local)))
        sftp.close()

        run(c, "chmod +x %s/run_fight.sh" % args.remote)
        run(c, "cd %s && python3 -m py_compile fight.py && echo PY_COMPILE_OK && ls -l" % args.remote)

        if not args.skip_check:
            run(c, "cd %s && bash run_fight.sh check" % args.remote)

        if args.probe:
            run(c, "cd %s && python3 fight.py probe" % args.remote, timeout=90)
    finally:
        c.close()
    print("==== 部署结束（%.1fs），狗未收到任何运动指令 ====" % (time.time() - t0))
    print()
    print("下一步（在狗上）：")
    print("  ssh ysc@%s" % args.host)
    print("  cd %s && bash run_fight.sh dry          # 干跑，不发指令" % args.remote)
    print("  cd %s && bash run_fight.sh fight        # 真打" % args.remote)


if __name__ == "__main__":
    main()
