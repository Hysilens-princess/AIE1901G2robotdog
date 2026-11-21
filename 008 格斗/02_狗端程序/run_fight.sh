#!/bin/bash
# =====================================================================
# Lite3 全自主格斗 · 狗端启动器（在运动主机 192.168.1.120 上跑）
#
#   bash run_fight.sh check          只体检，不动狗
#   bash run_fight.sh probe          相机/UDP/遥测体检（只读）
#   bash run_fight.sh calib 60       相机调参 60 秒（画面检测结果打屏）
#   bash run_fight.sh dry            干跑：完整流程 25 s，不发指令
#   bash run_fight.sh fight          真打（默认参数）
#   bash run_fight.sh stop           立刻停（SIGTERM -> 程序自己零速复位）
#   bash run_fight.sh log            看最近一次日志
#   bash run_fight.sh replay x.mp4   离线回放调参
#
# 常用环境变量：
#   PRESET=push|ram|gentle   DUR=180   DELAY=5   GAIT=medium|fast
#   CHAN=vel|axis            MAXV=0.55 PUSHV=0.75
#   EXTRA="--stand"  追加任意 fight.py 参数
#
# 例：PRESET=ram MAXV=0.8 PUSHV=1.0 bash run_fight.sh fight
# =====================================================================
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1
PY_BIN="python3"
SCRIPT="$DIR/fight.py"
export PYTHONUNBUFFERED=1

PRESET="${PRESET:-push}"
DUR="${DUR:-180}"
DELAY="${DELAY:-5}"
GAIT="${GAIT:-medium}"
CHAN="${CHAN:-vel}"
MAXV="${MAXV:-}"
PUSHV="${PUSHV:-}"
AXIS_VMAX="${AXIS_VMAX:-2.4}"
EXTRA="${EXTRA:-}"
CSV="${CSV:-1}"
VIDEO="${VIDEO:-0}"

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }

need_script() {
    if [ ! -f "$SCRIPT" ]; then
        red "fight.py 不在 $DIR —— 先部署（电脑上 deploy_fight.py）"
        exit 1
    fi
}

check_only() {
    echo "==================== 狗端体检（格斗版） ===================="
    echo "主机名    : $(hostname)"
    echo "目录      : $DIR"
    echo "时间      : $(date '+%F %T')"

    if command -v "$PY_BIN" >/dev/null 2>&1; then
        green "python3   : $($PY_BIN -V 2>&1)"
    else
        red   "python3   : 没有！无法运行"; return 1
    fi

    if $PY_BIN -c "import cv2" 2>/dev/null; then
        green "cv2       : $($PY_BIN -c 'import cv2;print(cv2.__version__)' 2>&1)"
    else
        red   "cv2       : 导入失败 —— 视觉程序跑不了（见操作手册 B 计划）"
    fi

    if pgrep -f "[j]y_exe" >/dev/null 2>&1; then
        green "jy_exe    : 在跑（PID $(pgrep -f "[j]y_exe" | head -1)）"
    else
        yellow "jy_exe    : 没在跑 —— 发什么都不动"
    fi

    if ss -lun 2>/dev/null | grep -q ':43893'; then
        green "43893     : 在监听"
    else
        yellow "43893     : ss 没看到（可能没 ss 命令，以 jy_exe 进程为准）"
    fi

    if [ -e /dev/video0 ]; then
        echo "摄像头    : $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
    else
        yellow "摄像头    : 没有 /dev/video* —— 相机没插上或被占了"
    fi
    echo "占用情况  : $(pgrep -af '[t]rack' | head -3 || true)"
    if [ -f /home/ysc/jy_exe/conf/flat_rl_config.yaml ]; then
        echo "速度上限  : $(grep -m1 max_velocity_target /home/ysc/jy_exe/conf/flat_rl_config.yaml)"
    fi
    echo "磁盘      : $(df -h "$DIR" | tail -1 | awk '{print $4" 可用"}')"
    echo "网络      : $(ip -4 -o addr show 2>/dev/null | awk '{print $2"="$4}' | tr '\n' ' ')"
    echo "=========================================================="
    return 0
}

cleanup_old() {
    OLD=$(pgrep -f "[f]ight\.py" 2>/dev/null)
    if [ -n "$OLD" ]; then
        yellow "发现上一轮残留：$(echo $OLD | tr '\n' ' ') —— 先清理（它发的速度帧会和新的一轮打架）"
        pkill -TERM -f "[f]ight\.py" 2>/dev/null
        for i in 1 2 3 4 5 6; do
            sleep 0.5
            [ -z "$(pgrep -f "[f]ight\.py" 2>/dev/null)" ] && break
        done
        if [ -n "$(pgrep -f "[f]ight\.py" 2>/dev/null)" ]; then
            red "SIGTERM 没停住，强杀（狗会在 1 s 内靠心跳超时自停）"
            pkill -9 -f "[f]ight\.py" 2>/dev/null
            sleep 0.4
        fi
        green "清场完成"
    else
        green "无残留进程"
    fi
}

MODE="${1:-help}"

case "$MODE" in
    check)
        check_only
        ;;

    probe)
        need_script
        $PY_BIN fight.py probe
        ;;

    calib)
        need_script
        SECS="${2:-0}"
        yellow "调参模式：狗要架起或趴着，镜头前晃对手的狗/你的手，看 bearing 和 area"
        $PY_BIN fight.py calib --seconds "$SECS" --dump-dir ./calib_frames
        ;;

    dry)
        need_script
        cleanup_old
        TS=$(date +%Y%m%d_%H%M%S); LOG="$DIR/logs/dry_${TS}.log"; mkdir -p "$DIR/logs"
        echo
        yellow "干跑 25 s：完整模式序列 + 状态机，但 --apply 没加，狗不会动"
        setsid nohup $PY_BIN -u fight.py run \
            --preset "$PRESET" --gait "$GAIT" --chan "$CHAN" \
            --duration 25 --auto-start --delay 3 --no-pause \
            --csv "$DIR/logs/dry_${TS}.csv" \
            --dump-dir "$DIR/logs/dry_frames_${TS}" --dump-every 20 \
            $EXTRA > "$LOG" 2>&1 < /dev/null &
        sleep 2
        tail -n 40 "$LOG"
        echo
        green "日志：$LOG"
        ;;

    fight)
        need_script
        check_only
        cleanup_old
        TS=$(date +%Y%m%d_%H%M%S)
        mkdir -p "$DIR/logs"
        LOG="$DIR/logs/fight_${TS}.log"
        ARGS="--preset $PRESET --gait $GAIT --chan $CHAN --axis-vmax $AXIS_VMAX \
--duration $DUR --auto-start --delay $DELAY --no-pause --apply"
        [ -n "$MAXV" ]  && ARGS="$ARGS --max-v $MAXV"
        [ -n "$PUSHV" ] && ARGS="$ARGS --push-v $PUSHV"
        [ "$CSV" = "1" ]   && ARGS="$ARGS --csv $DIR/logs/fight_${TS}.csv"
        [ "$VIDEO" = "1" ] && ARGS="$ARGS --video $DIR/logs/fight_${TS}.avi --dump-dir $DIR/logs/frames_${TS} --dump-every 30"
        ARGS="$ARGS $EXTRA"

        echo
        echo "============== 全自主格斗 开打 =============="
        echo " 预设      : $PRESET   (靠近 ${MAXV:-预设} / 顶 ${PUSHV:-预设} m/s)"
        echo " 通道      : $CHAN    (vel=0x0140 自主模式 / axis=摇杆 手柄模式)"
        echo " 步态      : $GAIT"
        echo " 余量      : ${DUR}s 后自动停；armed 后 ${DELAY}s 或看见目标即开打"
        echo " 日志      : $LOG"
        echo "============================================"
        echo

        setsid nohup $PY_BIN -u fight.py run $ARGS > "$LOG" 2>&1 < /dev/null &
        PID=$!
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            green "已启动 PID $PID（脱离会话，关窗口/拔网线都不影响）"
        else
            red "启动后立刻退出了，看日志："
        fi
        echo "---- 日志开头 ----"
        sed -n '1,40p' "$LOG" 2>/dev/null
        echo "------------------"
        echo
        yellow "停：bash run_fight.sh stop     看日志：bash run_fight.sh log"
        ;;

    stop)
        red "发 SIGTERM（程序会先零速再复位，别用 kill -9）"
        pkill -TERM -f "[f]ight\.py" 2>/dev/null
        sleep 2.0
        if pgrep -f "[f]ight\.py" >/dev/null 2>&1; then
            yellow "还没退，再来一次"
            pkill -TERM -f "[f]ight\.py" 2>/dev/null
            sleep 1.5
        fi
        if pgrep -f "[f]ight\.py" >/dev/null 2>&1; then
            red "仍没退：kill -9（狗 1 s 内靠心跳超时自停）"
        else
            green "已停"
        fi
        ;;

    log)
        f=$(ls -t "$DIR"/logs/*.log 2>/dev/null | head -1)
        [ -z "${f:-}" ] && { red "没有日志"; exit 1; }
        echo "== $f =="
        tail -n 60 "$f"
        ;;

    replay)
        need_script
        [ -z "${2:-}" ] && { red "用法：bash run_fight.sh replay /path/x.mp4"; exit 1; }
        mkdir -p "$DIR/logs"
        $PY_BIN fight.py replay --video "$2" \
            --video-out "${DIR}/logs/replay_out.avi" \
            --csv "${DIR}/logs/replay_out.csv"
        ;;

    selftest)
        need_script
        $PY_BIN fight.py selftest
        ;;

    *)
        sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
