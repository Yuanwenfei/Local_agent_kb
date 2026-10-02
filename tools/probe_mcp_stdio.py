"""临时探针：按 mcp.json 的方式拉起 kb_mcp_server.py，跑一次 MCP 握手 + tools/list。

用来验证「MCP error -32000: Connection closed」是否已修复（即服务端能完整应答一个会话）。
"""
import json
import os
import subprocess
import sys
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(PROJECT_ROOT, "python312", "python.exe")
SRV = os.path.join(PROJECT_ROOT, "kb_mcp_server.py")
if not os.path.exists(EXE):  # 非常规布局时退回当前解释器
    EXE = sys.executable

env = dict(os.environ, KB_USE_GPU="0", QDRANT_HOST="localhost", QDRANT_PORT="6333",
           KB_COLLECTION="emulate3d_docs", KB_IDLE_TIMEOUT="600",
           PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

p = subprocess.Popen([EXE, SRV], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, cwd="C:\\", env=env,
                     text=True, encoding="utf-8", bufsize=1)

got: dict = {}
lock = threading.Lock()
done = threading.Event()


def reader():
    for line in p.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        with lock:
            if "id" in msg:
                got[msg["id"]] = msg
        if len([k for k in got if k in (1, 2)]) >= 2:
            done.set()
            return


threading.Thread(target=reader, daemon=True).start()


def send(obj):
    p.stdin.write(json.dumps(obj) + "\n")
    p.stdin.flush()


send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "probe", "version": "1"}}})
send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})

ok = done.wait(timeout=90)
try:
    p.terminate()
except Exception:
    pass

if not ok:
    print("FAIL: 90 秒内未收齐 initialize / tools/list 应答（进程退出码 %s）" % p.poll())
    sys.exit(1)

init = got.get(1, {})
tools = [t["name"] for t in got.get(2, {}).get("result", {}).get("tools", [])]
print("initialize:", init.get("result", {}).get("serverInfo", {}))
print("tools 数量:", len(tools))
print("关键工具:", [t for t in tools if t in ("search_tech_kb", "kb_caption", "kb_meta_lint", "kb_dupes")])
print("OK: MCP 握手与 tools/list 均正常")
