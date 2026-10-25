#!/usr/bin/env bash
# Source after release-common.sh. Callbacks run in the caller's shell context.
build_run_leg() (
  local leg="$1" command="$2"
  set +m
  # Keep completion separate from the worker's private EXIT/cleanup traps.
  trap 'status=$?; printf "%s %s\n" "$leg" "$status" >&8' EXIT
  "$command"
)

build_supervise_cleanup() {
  local pid callback
  trap - EXIT INT TERM
  for pid in ${build_leg_pids[@]+"${build_leg_pids[@]}"}; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for callback in ${build_leg_cancellations[@]+"${build_leg_cancellations[@]}"}; do
    "$callback" || printf 'WARNING: %s cancellation failed; inspect %s\n' "$build_context" "$build_log_dir" >&2
  done
  # Stop descendants as well as leaders, allowing bounded time for EXIT traps.
  sleep 2
  for pid in ${build_leg_pids[@]+"${build_leg_pids[@]}"}; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exec 8>&-
  rm -f "$build_log_dir/completions"
}

# context, log directory, sequential (0/1), then each leg's name, command,
# cancellation callback and log filename. Omit the second leg for single-role runs.
build_supervise() {
  local build_context="$1" build_log_dir="$2" sequential="$3"
  local first="$4" first_command="$5" first_cancel="$6" first_log="$7"
  local second="${8:-}" second_command="${9:-}" second_cancel="${10:-}" second_log="${11:-}"
  local completed status pid active_pid remaining=0 first_pid="" second_pid=""
  local -a build_leg_pids=() build_leg_cancellations=() active_pids=()
  mkdir -p "$build_log_dir"
  printf '[%s] log: %s/%s\n' "$first" "$build_log_dir" "$first_log"
  [[ -z "$second" ]] || printf '[%s] log: %s/%s\n' "$second" "$build_log_dir" "$second_log"
  mkfifo "$build_log_dir/completions"
  exec 8<>"$build_log_dir/completions"
  trap build_supervise_cleanup EXIT
  trap 'echo "$build_context interrupted (INT); stopping both legs" >&2; exit 130' INT
  trap 'echo "$build_context interrupted (TERM); stopping both legs" >&2; exit 143' TERM
  # Workers have private process groups. The foreground supervisor observes either
  # completion immediately, rather than blocking on the primary leg's build.
  set -m
  if [[ "$sequential" == 0 && -n "$second" ]]; then
    build_run_leg "$second" "$second_command" >"$build_log_dir/$second_log" 2>&1 &
    second_pid=$!
    build_leg_pids+=("$second_pid")
    build_leg_cancellations+=("$second_cancel")
    remaining=1
  fi
  build_run_leg "$first" "$first_command" >"$build_log_dir/$first_log" 2>&1 &
  first_pid=$!
  build_leg_pids+=("$first_pid")
  build_leg_cancellations+=("$first_cancel")
  remaining=$((remaining + 1))
  set +m
  while ((remaining)); do
    # Poll with read -t 0, which consumes nothing. A timed read that expires
    # mid-line keeps its partial input, so the next read saw "xpert" for "expert".
    # Each leg writes its whole line in one printf, so a ready line is complete.
    if read -r -t 0 <&8; then
      read -r completed status <&8
    else
      # SIGKILL/OOM skips EXIT notification. Do not wait forever on a FIFO whose
      # write end the supervisor itself holds open; reap a dead worker instead.
      completed=""
      for pid in "${build_leg_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
          [[ "$pid" != "$first_pid" ]] && completed="$second" || completed="$first"
          break
        fi
      done
      [[ -n "$completed" ]] || { sleep 1; continue; }
      # The poll above can run an instant before a worker's notification becomes
      # visible, and the notice is written before the worker exits: nothing writes
      # after exit, so give a dead worker's line a moment to surface before
      # misreporting a normal completion as a kill. A killed worker never writes,
      # so this read just times out before the diagnosis.
      if read -r -t 1 late_leg late_status <&8; then
        completed="$late_leg"
        status="$late_status"
      else
        status=0
        wait "$pid" || status=$?
        release_die "[$completed] $build_context leg failed (exit $status); killed without exit status notification; stopping the other leg; see $build_log_dir"
      fi
    fi
    case "$completed" in
      "$first") pid="$first_pid" ;;
      "$second") pid="$second_pid" ;;
      *) release_die "invalid build leg completion: $completed" ;;
    esac
    wait "$pid" || true
    if [[ "$status" != 0 ]]; then
      if [[ "$completed" == "$first" ]]; then
        tail -n 20 "$build_log_dir/$first_log" >&2
      else
        tail -n 20 "$build_log_dir/$second_log" >&2
      fi
      release_die "[$completed] $build_context leg failed (exit $status); stopping the other leg; see $build_log_dir"
    fi
    # Do not retain completed group IDs: another job could later reuse the ID.
    active_pids=()
    for active_pid in "${build_leg_pids[@]}"; do
      [[ "$active_pid" == "$pid" ]] || active_pids+=("$active_pid")
    done
    build_leg_pids=("${active_pids[@]}")
    printf '[%s] %s leg complete\n' "$completed" "$build_context"
    remaining=$((remaining - 1))
    if [[ "$sequential" == 1 && "$completed" == "$first" && -n "$second" ]]; then
      set -m
      build_run_leg "$second" "$second_command" >"$build_log_dir/$second_log" 2>&1 &
      second_pid=$!
      build_leg_pids+=("$second_pid")
      build_leg_cancellations+=("$second_cancel")
      remaining=$((remaining + 1))
      set +m
    fi
  done
  trap - EXIT INT TERM
  exec 8>&-
  rm -f "$build_log_dir/completions"
}
