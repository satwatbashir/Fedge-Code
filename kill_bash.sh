#!/bin/bash
# kill_bash_enhanced.sh — Complete Fedge/Mininet environment reset with SAFE deep cleanup
# Usage:
#   sudo bash kill_bash_enhanced.sh            # full wipe (recommended for fresh start)
#   sudo bash kill_bash_enhanced.sh --preserve-data   # keep rounds/ and metrics/

set +e
umask 022

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

PRESERVE_DATA=false
if [[ "$1" == "--preserve-data" ]]; then
  PRESERVE_DATA=true
  echo -e "${YELLOW}⚠️  Preserve mode: keeping rounds/ and metrics/${NC}"
fi

is_root() { [[ "$(id -u)" -eq 0 ]]; }
run_root() { if is_root; then "$@"; else sudo "$@"; fi; }

# Real caller's home (even under sudo)
CALLER="${SUDO_USER:-$USER}"
if [[ "$CALLER" == "root" ]]; then CALLER_HOME="/root"; else CALLER_HOME="/home/$CALLER"; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

print_header() {
  echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║  Fedge / Flower / Mininet — SAFE Full Cleanup & Reset      ║${NC}"
  echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
  echo ""
}

safe_rm_dir() {
  local d="$1"
  if [[ "$PWD" == "$SCRIPT_DIR" && -d "$d" ]]; then
    rm -rf "$d" 2>/dev/null || run_root rm -rf "$d"
    echo "  ✓ Removed $d/"
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

print_header

echo -e "${YELLOW}🔪 Step 1: Killing Fedge/Flower/Mininet processes...${NC}"
PATTERNS=(
  "cloud_flower\.py" "leaf_server\.py" "leaf_client\.py" "proxy_client\.py"
  "orchestrator\.py" "(^|/)tools/net_topo\.py" "(^|/)net_topo\.py"
  "flower\.server" "flwr\.server" "mininet"
)

# Targeted: pgrep searches COMMAND, not username; exclude self
for pat in "${PATTERNS[@]}"; do
  # -f: match full cmdline; -a: print cmd too (for logs)
  mapfile -t PIDS < <(pgrep -fa "$pat" | awk -v self=$$ -v ppid=$PPID '{print $1}' | grep -Ev "^($$|$PPID)$" || true)
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    printf "  → %s\n" "Killing $(printf '%s ' "${PIDS[@]}") for /$pat/"
    run_root kill -9 "${PIDS[@]}" 2>/dev/null || true
  fi
done

# Mininet quick clean
have_cmd mn && run_root mn -c >/dev/null 2>&1

# Ports to clear
PORTS=(5000 5001 5002 5003 5004 5005 6000 6100 6200)
for port in "${PORTS[@]}"; do
  if have_cmd lsof; then
    PIDS=$(lsof -ti:"$port" 2>/dev/null || true)
  else
    PIDS=$(ss -lptn 2>/dev/null | awk -v p=":$port" '$0~p{match($0,/pid=([0-9]+)/,m); if(m[1]!="") print m[1]}' || true)
  fi
  [[ -n "$PIDS" ]] && run_root kill -9 $PIDS 2>/dev/null && echo "  ✓ Killed process on port $port"
done

# Kill screen/tmux sessions containing our processes (by window titles/commands)
if have_cmd screen; then
  screen -ls 2>/dev/null | awk '/\t/ {print $1}' | while read -r sess; do
    screen -S "$sess" -Q windows 2>/dev/null | grep -E "flower|flwr|mininet|leaf_|cloud_|proxy_|orchestrator|net_topo" >/dev/null 2>&1 \
      && screen -S "$sess" -X quit 2>/dev/null && echo "  ✓ Killed screen session: $sess"
  done
fi
if have_cmd tmux; then
  tmux ls 2>/dev/null | awk -F: '{print $1}' | while read -r sess; do
    tmux list-panes -t "$sess" -F '#{pane_current_command} #{pane_current_path}' 2>/dev/null \
      | grep -E "flower|flwr|mininet|leaf_|cloud_|proxy_|orchestrator|net_topo" >/dev/null 2>&1 \
      && tmux kill-session -t "$sess" 2>/dev/null && echo "  ✓ Killed tmux session: $sess"
  done
fi

# Kill Ray by process name (not regex string)
pgrep -f "ray::" >/dev/null 2>&1 && run_root pkill -9 -f "ray::" && echo "  ✓ Killed Ray actors"

sleep 1
echo -e "${GREEN}  ✓ Process cleanup complete${NC}"
echo ""

echo -e "${YELLOW}🧹 Step 1.5: Deep system resource cleanup...${NC}"
if have_cmd ip; then
  echo "  → Removing network namespaces..."
  for ns in $(run_root ip netns list 2>/dev/null | awk '{print $1}'); do
    run_root ip netns delete "$ns" 2>/dev/null && echo "  ✓ Removed netns: $ns"
  done
fi

if have_cmd ipcs; then
  echo "  → Cleaning IPC resources..."
  ME="$(id -un)"
  ipcs -m 2>/dev/null | awk -v me="$ME" '$3==me{print $2}' | xargs -r -n1 ipcrm -m 2>/dev/null
  ipcs -s 2>/dev/null | awk -v me="$ME" '$3==me{print $2}' | xargs -r -n1 ipcrm -s 2>/dev/null
  ipcs -q 2>/dev/null | awk -v me="$ME" '$3==me{print $2}' | xargs -r -n1 ipcrm -q 2>/dev/null
fi

echo "  → Cleaning shared memory (/dev/shm)..."
rm -rf /dev/shm/fedge_* /dev/shm/flwr_* /dev/shm/flower_* 2>/dev/null || true

echo "  → Cleaning PID/socket/core files..."
rm -f /tmp/*.pid /var/run/fedge*.pid /tmp/fedge*.pid 2>/dev/null || true
find /tmp -maxdepth 1 -user "$CALLER" -name "*.sock" -delete 2>/dev/null || true
rm -f core.* /tmp/core.* 2>/dev/null || true

echo -e "${GREEN}  ✓ Deep system resource cleanup complete${NC}"
echo ""

echo -e "${YELLOW}🧹 Step 2: Cleaning Mininet/OVS/tc/veth state...${NC}"
have_cmd mn && run_root mn -c >/dev/null 2>&1 && echo "  ✓ mn -c" || true

if have_cmd ovs-vsctl; then
  for br in $(run_root ovs-vsctl list-br 2>/dev/null); do
    run_root ovs-vsctl del-br "$br" 2>/dev/null && echo "  ✓ Removed OVS bridge: $br"
  done
fi

# Determine primary/SSH iface; don't touch its qdisc to avoid dropping your session
ACTIVE_IFACE=""
if have_cmd ip; then
  ACTIVE_IFACE=$(ip route get 1.1.1.1 2>/dev/null | awk '/dev/{for(i=1;i<=NF;i++)if($i=="dev"){print $(i+1);exit}}')
fi
[[ -n "$SSH_CONNECTION" && -z "$ACTIVE_IFACE" ]] && ACTIVE_IFACE=$(echo "$SSH_CONNECTION" | awk '{print $5}')

echo "  → Removing tc qdiscs from non-primary interfaces..."
if have_cmd tc; then
  for iface in $(ls /sys/class/net 2>/dev/null); do
    [[ "$iface" == "lo" || "$iface" == "$ACTIVE_IFACE" ]] && continue
    run_root tc qdisc del dev "$iface" root   2>/dev/null || true
    run_root tc qdisc del dev "$iface" ingress 2>/dev/null || true
  done
fi

echo "  → Removing veth pairs..."
if have_cmd ip; then
  ip -o link show 2>/dev/null | awk -F': ' '$2 ~ /^veth/ {print $2}' | while read -r v; do
    run_root ip link delete "$v" 2>/dev/null && echo "  ✓ Removed veth: $v"
  done
  # GRE tunnels (safe)
  ip -o link show type gre 2>/dev/null | awk -F': ' '{print $2}' | sed 's/@.*//' | while read -r t; do
    run_root ip link delete "$t" 2>/dev/null && echo "  ✓ Removed GRE tunnel: $t"
  done
fi
echo -e "${GREEN}  ✓ Network cleanup complete${NC}"
echo ""

echo -e "${YELLOW}🧠 Step 3: Dropping filesystem caches...${NC}"
echo "  Memory before:"; free -h 2>/dev/null | grep -E "Mem:|Swap:" | sed 's/^/    /'
sync
run_root sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "  ⚠ Need root to drop caches"
sleep 1
echo "  Memory after:";  free -h 2>/dev/null | grep -E "Mem:|Swap:" | sed 's/^/    /'
echo -e "${GREEN}  ✓ Cache cleanup complete${NC}"
echo ""

echo -e "${YELLOW}🧾 Step 4: Removing logs, temp files, and project artifacts...${NC}"
echo "  → Removing logs..."
rm -f *.log *.log.err cloud*.log server*.log client*.log proxy*.log c*_*.log 2>/dev/null || true
run_root rm -rf logs/ 2>/dev/null || true

echo "  → Removing signal files..."
safe_rm_dir signals

echo "  → Removing model checkpoints..."
safe_rm_dir models

if [[ "$PRESERVE_DATA" == "false" ]]; then
  echo "  → Removing rounds/ and metrics/ (fresh start)..."
  safe_rm_dir rounds
  safe_rm_dir metrics
else
  echo -e "  ${BLUE}⚠️  Preserving rounds/ and metrics/${NC}"
fi

echo "  → Removing Python caches & temp..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
run_root rm -rf /tmp/fedge_* /tmp/flwr_* /tmp/ray* /tmp/flower_* /tmp/mininet_* 2>/dev/null || true

# Clean Ray for both caller and root
rm -rf "$CALLER_HOME/.ray" "$CALLER_HOME/ray_results" 2>/dev/null || true
run_root rm -rf /root/.ray /root/ray_results 2>/dev/null || true

echo -e "${GREEN}  ✓ File cleanup complete${NC}"
echo ""

echo -e "${YELLOW}🔍 Step 5: Checking for lingering processes and open files...${NC}"
LINGERING=$(pgrep -fa "flower|flwr|mininet|leaf_|cloud_|proxy_|orchestrator\.py|net_topo\.py" || true)
if [[ -n "$LINGERING" ]]; then
  echo -e "${RED}  ⚠️  Some processes still visible:${NC}"
  echo "$LINGERING" | sed 's/^/    /'
else
  echo -e "${GREEN}  ✓ No lingering processes found${NC}"
fi

if have_cmd lsof; then
  echo "  → Checking for open files..."
  OPEN_FILES=$(lsof -u "$CALLER" 2>/dev/null | grep -E "(flower|fedge|mininet)" || true)
  if [[ -n "$OPEN_FILES" ]]; then
    echo -e "${RED}  ⚠️  Some files still open (showing first 5):${NC}"
    echo "$OPEN_FILES" | head -5 | sed 's/^/    /'
  else
    echo -e "${GREEN}  ✓ No open files found${NC}"
  fi
fi

if have_cmd ss; then
  echo "  → Checking for established connections..."
  CONNECTIONS=$(ss -antp 2>/dev/null | grep -E "(flower|fedge|mininet)" || true)
  [[ -n "$CONNECTIONS" ]] && echo -e "${RED}  ⚠️  Some connections still active:${NC}" && echo "$CONNECTIONS" | sed 's/^/    /' || echo -e "${GREEN}  ✓ No active connections found${NC}"
fi
echo ""

echo -e "${YELLOW}🔌 Step 6: Verifying port availability...${NC}"
PORTS_IN_USE=false
for port in "${PORTS[@]}"; do
  if have_cmd lsof && lsof -ti:$port >/dev/null 2>&1; then
    PID=$(lsof -ti:$port 2>/dev/null | head -1)
    PNAME=$(ps -p "$PID" -o comm= 2>/dev/null || echo "unknown")
    echo -e "  ${RED}✗ Port $port in use by PID $PID ($PNAME)${NC}"; PORTS_IN_USE=true
  elif have_cmd ss && ss -ltn 2>/dev/null | grep -q ":$port "; then
    echo -e "  ${RED}✗ Port $port still in use${NC}"; PORTS_IN_USE=true
  else
    echo -e "  ${GREEN}✓ Port $port available${NC}"
  fi
done
[[ "$PORTS_IN_USE" == "true" ]] && echo -e "${YELLOW}  Re-run cleanup or manually kill processes on those ports${NC}"
echo ""

echo -e "${YELLOW}🔍 Step 7: Final verification summary...${NC}"
PROC_COUNT=$(pgrep -f "flower|flwr|mininet|leaf_|cloud_|proxy_|orchestrator\.py|net_topo\.py" | wc -l)
echo "  • Matching processes: $PROC_COUNT"

PORT_COUNT=0
for port in "${PORTS[@]}"; do
  if (have_cmd lsof && lsof -ti:$port >/dev/null 2>&1) || (have_cmd ss && ss -ltn 2>/dev/null | grep -q ":$port "); then
    ((PORT_COUNT++))
  fi
done
echo "  • Ports in use: $PORT_COUNT/${#PORTS[@]}"

if have_cmd ipcs; then
  ME="$(id -un)"
  SHM_COUNT=$(ipcs -m 2>/dev/null | awk -v me="$ME" '$3==me' | wc -l)
  SEM_COUNT=$(ipcs -s 2>/dev/null | awk -v me="$ME" '$3==me' | wc -l)
  MSG_COUNT=$(ipcs -q 2>/dev/null | awk -v me="$ME" '$3==me' | wc -l)
  echo "  • IPC resources (yours): $SHM_COUNT shm, $SEM_COUNT sem, $MSG_COUNT msg"
fi

if have_cmd ip; then
  NS_COUNT=$(run_root ip netns list 2>/dev/null | wc -l)
  echo "  • Network namespaces: $NS_COUNT"
fi
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    SAFE Cleanup Summary                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo -e "${GREEN}✓ Processes terminated (targeted)${NC}"
echo -e "${GREEN}✓ Net namespaces & IPC cleaned${NC}"
echo -e "${GREEN}✓ Network state cleared (mn/OVS/tc/veth/tunnels; primary iface preserved)${NC}"
echo -e "${GREEN}✓ Caches dropped${NC}"
echo -e "${GREEN}✓ PID/socket/core files removed${NC}"
echo -e "${GREEN}✓ Logs/temp/artifacts removed${NC}"
if [[ "$PRESERVE_DATA" == "true" ]]; then
  echo -e "${YELLOW}⚠️  Data preserved: rounds/, metrics/${NC}"
else
  echo -e "${GREEN}✓ Data wiped: signals/, models/, rounds/, metrics/${NC}"
fi

echo ""
if [[ $PROC_COUNT -eq 0 && $PORT_COUNT -eq 0 ]]; then
  echo -e "${GREEN}✅ CLEANUP COMPLETE: Environment is ready for a new experiment.${NC}"
else
  echo -e "${YELLOW}⚠️  PARTIAL CLEANUP: Some resources still active. You may run:${NC}"
  echo -e "${YELLOW}   sudo killall -9 python python3${NC}"
fi
echo ""

