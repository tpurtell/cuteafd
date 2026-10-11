#!/usr/bin/env bash
# Independent, loopback-only DSH lifecycle. No model deployment or hardware locks.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
action="${1:-status}"
[[ $# -le 1 ]] || { printf 'Usage: agent.sh start|stop|status\n' >&2; exit 2; }
name="${CUTEAFD_AGENT_CONTAINER:-cuteafd-agent}"
image="${CUTEAFD_AGENT_IMAGE:-cuteafd-agent:wip}"
port="${CUTEAFD_AGENT_PORT:-3010}"
[[ "$port" =~ ^[0-9]+$ && "$port" -gt 0 && "$port" -le 65535 ]] || { printf 'Invalid agent port\n' >&2; exit 2; }
case "$action" in
  -h|--help)
    printf '%s\n' 'Usage: agent.sh start|stop|status' \
      'Reads the coordinator cuteafd.config (CUTEAFD_AGENT_CONFIG overrides).' \
      'Uses its API_KEY_FILE and ADDR; refuses if the coordinator key cannot be resolved.' \
      'Set CUTEAFD_AGENT_PUBLIC_URL=https://HOST:PORT/agent/app/ when visiting the dashboard' \
      'through a different hostname or reverse proxy; the trusted authority is derived from it.' \
      'Overrides: CUTEAFD_AGENT_IMAGE, CUTEAFD_AGENT_PORT, CUTEAFD_AGENT_HOME,' \
      'CUTEAFD_AGENT_MODEL, CUTEAFD_AGENT_GATEWAY_URL, CUTEAFD_AGENT_DISCOVERY_URL.'
    ;;
  status) docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null || { printf 'stopped\n'; exit 1; } ;;
  stop) docker stop --time 15 "$name" >/dev/null 2>&1 || true; docker rm "$name" >/dev/null 2>&1 || true ;;
  start)
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" == true ]]; then
      printf 'Agent sidecar already running\n'; exit 0
    fi
    home="${CUTEAFD_AGENT_HOME:-$HOME/.local/share/cuteafd/agent}"
    umask 077
    mkdir -p "$home"; chmod 700 "$home"
    # Serialize first-start credentials and lifecycle without using a hardware lock.
    exec 9>"$home/launcher.lock"; flock -w 30 9
    # A second start may have completed while we waited for the lifecycle lock.
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null || true)" == true ]]; then
      printf 'Agent sidecar already running\n'; exit 0
    fi
    source "$repo_root/scripts/lib/release-common.sh"
    release_load_config "${CUTEAFD_AGENT_CONFIG:-$repo_root/cuteafd.config}" stop
    release_resolve_api_key "${ENABLE_BENCH:-off}" "${INSTANCE:-default}"
    [[ -n "${API_KEY_FILE:-}" && -r "$API_KEY_FILE" ]] || {
      printf 'Cannot determine coordinator API key: configure API_KEY_FILE in the coordinator config or environment\n' >&2; exit 2;
    }
    export API_KEY_FILE
    # Wildcard listeners have no browser hostname; use localhost until overridden.
    defaults="$(python3 - "$ADDR" "${CUTEAFD_AGENT_PUBLIC_URL:-}" "${CUTEAFD_AGENT_GATEWAY_URL:-}" <<'PY_URL'
import sys, urllib.parse
addr, public, gateway_override = sys.argv[1:]
url = urllib.parse.urlsplit(public or "http://" + addr)
if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password:
    raise SystemExit("Invalid agent public URL/listen address")
host = url.hostname
if host in ("0.0.0.0", "::"): host = "localhost"
if ":" in host: host = "[" + host + "]"
port = url.port or (443 if url.scheme == "https" else 80)
authority = host + (":" + str(port) if port != (443 if url.scheme == "https" else 80) else "")
print(url.scheme + "://" + authority + "/agent/app/")
print(authority)
listen = urllib.parse.urlsplit("http://" + addr)
listen_host = listen.hostname
listen_port = listen.port or 8000
if listen_host in ("127.0.0.1", "localhost", "::1") and not gateway_override:
    raise SystemExit("Loopback-only coordinator is unreachable from sidecar bridge: set CUTEAFD_AGENT_GATEWAY_URL to a reachable endpoint or bind coordinator to 0.0.0.0")
if listen_host in ("0.0.0.0", "::"):
    discovery_host, gateway_host = "127.0.0.1", "host.docker.internal"
else:
    discovery_host = gateway_host = listen_host
    if ":" in listen_host: discovery_host = gateway_host = "[" + listen_host + "]"
print("http://" + discovery_host + ":" + str(listen_port))
print("http://" + gateway_host + ":" + str(listen_port))
PY_URL
)"
    mapfile -t agent_defaults <<<"$defaults"
    export CUTEAFD_AGENT_PUBLIC_URL="${CUTEAFD_AGENT_PUBLIC_URL:-${agent_defaults[0]}}"
    export CUTEAFD_AGENT_AUTHORITY="${CUTEAFD_AGENT_AUTHORITY:-${agent_defaults[1]}}"
    export CUTEAFD_AGENT_DISCOVERY_URL="${CUTEAFD_AGENT_DISCOVERY_URL:-${agent_defaults[2]}}"
    export CUTEAFD_AGENT_GATEWAY_URL="${CUTEAFD_AGENT_GATEWAY_URL:-${agent_defaults[3]}}"
    python3 "$repo_root/scripts/launch/agent-config.py" "$home"
    # Stdin pipe only: no secret in environment, arguments, shell expansion or logs.
    python3 "$repo_root/scripts/launch/agent-config.py" "$home" --emit-key |
      docker run --rm -i --user "$(id -u):$(id -g)" \
        --mount "type=bind,src=$(readlink -f "$home"),dst=/agent" \
        --env HOME=/agent/home --env DSH_HOME=/agent/dsh --env DSH_AGENTS_HOME=/agent/agents \
        --env DSH_TELEMETRY_DISABLED=1 --env DSH_REMOTE_ONLY=1 \
        --entrypoint /usr/local/bin/dsh-cuteafd "$image" set-credential CUTEAFD_AGENT_API_KEY
    docker rm "$name" >/dev/null 2>&1 || true
    # The image resolves its own container IPv4; DSH rejects wildcard binds.
    docker run -d --name "$name" --init --user "$(id -u):$(id -g)" \
      --publish "127.0.0.1:$port:3010" --add-host host.docker.internal:host-gateway \
      --mount "type=bind,src=$(readlink -f "$home"),dst=/agent" \
      --env HOME=/agent/home --env DSH_HOME=/agent/dsh --env DSH_AGENTS_HOME=/agent/agents \
      --env DSH_TELEMETRY_DISABLED=1 --env DSH_REMOTE_ONLY=1 \
      "$image" --profile cuteafd --port 3010 --no-open \
      --public-url "${CUTEAFD_AGENT_PUBLIC_URL:-http://localhost:8000/agent/app/}" \
      --trusted-host "${CUTEAFD_AGENT_AUTHORITY:-localhost:8000}" >/dev/null
    printf 'Agent sidecar started on loopback port %s\n' "$port"
    ;;
  *) printf 'Usage: agent.sh start|stop|status\n' >&2; exit 2 ;;
esac
