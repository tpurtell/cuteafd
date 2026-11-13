#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Two-GPU exchange over peer memory: SM-issued pushes with a release flag and
// a spinning acquire wait, no host involvement, graph-capturable (sequence
// numbers live in device memory, so replays advance them).
//
// A link is one direction A -> B: B owns `flag` (written by A's pushes) and
// `recv` (the payload); A owns `send_state` (u32 [2]: sequence, arrival
// counter); B owns `recv_state` (u32 [1]: last sequence waited for). All zeroed
// before first use. Peer access to B must be enabled on A.

// Loads the exchange kernels on the current device (call once per device
// before any wait is queued: a lazily loaded kernel's first launch may wait for
// the device to idle, which a spinning wait never lets it do).
int32_t cuteafd_peer_exchange_initialize(void);
// `out = bf16(a + b)` over `count` BF16 elements (FP32 add, round to nearest
// even: the same bits whichever order a GPU passes the two partials in).
// `out` must not overlap `a` or `b`.
int32_t cuteafd_peer_add_bf16_async(const void* a, const void* b, void* out, uint64_t count, void* stream);

// On A's stream: copy `bytes` (16-byte aligned, at most 2^40) from local
// `source` to peer `destination`, then publish the next sequence to the peer
// `flag` once every block's stores are visible system-wide. `blocks` 0 picks
// a default. The source must not be rewritten until the peer has waited.
int32_t cuteafd_peer_push_signal(void* destination, const void* source, uint64_t bytes,
    uint32_t* flag, uint32_t* send_state, uint32_t blocks, void* stream);
// A strided byte plane: `rows` rows of `row_bytes` from `source` (rows
// `source_pitch` apart) to `destination` (`destination_pitch` apart), either
// pointer on the current device or its peer. All five sizes and both pointers
// 16-byte aligned; a plane of 0 rows copies nothing.
typedef struct cuteafd_peer_plane {
  void* destination;
  const void* source;
  uint64_t rows;
  uint64_t row_bytes;
  uint64_t destination_pitch;
  uint64_t source_pitch;
} cuteafd_peer_plane_t;
#define CUTEAFD_PEER_MAX_PLANES 4
// On A's stream: copy `count` (1..4) planes in one launch, then publish the
// next sequence to `flag` as `cuteafd_peer_push_signal` does (same
// `send_state` layout, so the two may share a link). `flag` and `send_state`
// both null: copy only. Graph-capturable (the planes are kernel arguments).
int32_t cuteafd_peer_push_planes(const cuteafd_peer_plane_t* planes, uint32_t count, uint32_t* flag,
    uint32_t* send_state, uint32_t blocks, void* stream);
// On B's stream: one warp spins until `flag` reaches the next expected
// sequence (acquire), so later work on the stream sees the pushed bytes. After
// 60 s without it (the peer's stream failed or was never fed) the kernel traps:
// the stream faults instead of hanging. Abortable waits never trap (below).
int32_t cuteafd_peer_wait(const uint32_t* flag, uint32_t* recv_state, void* stream);

// Opt-in peer issue diagnostics. Records belong to the exchange, one per
// rank/slot, zeroed and retained through every captured replay. A host issue
// identity is immutable in a graph; it is NOT the runtime device sequence.
// Watchdog dumps are concurrent snapshots, not an atomic view of both ranks.
typedef struct cuteafd_peer_progress {
  uint64_t push_issue;
  uint64_t published_issue;
  uint64_t wait_issue;
  uint32_t published;
  uint32_t observed;
  uint32_t expected;
  uint32_t push_phase;  // 1 entered, 2 published
  uint32_t wait_phase;  // 3 waiting, 4 received, 5 aborted
  uint32_t reserved;
} cuteafd_peer_progress_t;
typedef struct cuteafd_peer_diagnostic {
  cuteafd_peer_progress_t* local;
  const cuteafd_peer_progress_t* peer;
  const uint32_t* local_control;
  const uint32_t* peer_control;
  uint64_t issue;
  uint64_t host_stream;
  uint32_t rank;
  uint32_t slot;
  uint32_t slots;
} cuteafd_peer_diagnostic_t;
int32_t cuteafd_peer_diagnostic_initialize(void);
int32_t cuteafd_peer_push_signal_diagnostic(void* destination, const void* source, uint64_t bytes,
    uint32_t* flag, uint32_t* send_state, uint32_t blocks, void* stream,
    const cuteafd_peer_diagnostic_t* diagnostic);
int32_t cuteafd_peer_push_planes_diagnostic(const cuteafd_peer_plane_t* planes, uint32_t count,
    uint32_t* flag, uint32_t* send_state, uint32_t blocks, void* stream,
    const cuteafd_peer_diagnostic_t* diagnostic);
// Null aborted selects the legacy (non-abortable) wait semantics.
int32_t cuteafd_peer_wait_diagnostic(const uint32_t* flag, uint32_t* recv_state,
    const uint32_t* aborted, void* stream, const cuteafd_peer_diagnostic_t* diagnostic);

// Host proxy mailbox (the device-driven Spark exchange): on the stream, after
// its earlier work (D2H copies into the pinned mailbox), write the four
// `words` to `descriptor` and publish the next sequence to `flag` (both
// pinned, device-mapped host memory) with a system-scope release; the proxy
// thread spinning on `flag` then reads the mailbox. `send_state` (u32 [1],
// device memory, zeroed) carries the sequence across graph replays. The
// proxy answers through another pinned flag that `cuteafd_peer_wait` spins on
// (host-mapped flags work there too).
int32_t cuteafd_host_signal(uint32_t* flag, uint32_t* send_state, uint32_t* descriptor,
    const uint32_t* words, void* stream);

// Optional terminal cancellation. The legacy wait/symbol above is unchanged.
// Initialize before enqueueing any abortable wait. `aborted` is a separate,
// zeroed local u32 that outlives every eager/captured wait and is never reset.
// Cancellation skips a wait without changing its receive sequence. Afterwards
// the owner must drain all streams, reject reuse, and retain buffers if draining
// fails. Publish on an independent stream so a blocked wait cannot prevent it.
int32_t cuteafd_peer_abort_initialize(void);
int32_t cuteafd_peer_wait_abortable(const uint32_t* flag, uint32_t* recv_state,
    const uint32_t* aborted, void* stream);
int32_t cuteafd_peer_abort_publish(uint32_t* aborted, void* independent_stream);
// An abortable wait (non-null `aborted`) that times out (60 s without its push)
// never traps once `cuteafd_peer_abort_initialize` ran on its device: it records the timeout in
// one process-wide mapped host record, publishes its own `aborted` word and
// exits without advancing its receive sequence; every other abortable wait on
// any device then exits too. Returns 1 once a timeout was recorded (0: none;
// the record is sticky), with its expected/observed sequence and flag address.
// The owner must treat it as terminal: reject reuse and drain.
int32_t cuteafd_peer_wait_timeout(uint32_t* expected, uint32_t* observed, uint64_t* flag);
// Test-only: the abortable wait timeout of the current device (default 60 s).
int32_t cuteafd_peer_wait_timeout_configure(uint64_t timeout_ns);

// On the stream: copy `count` (1..4) byte regions (`bytes[i]` from
// `sources[i]` to `destinations[i]`, any alignment; device pointers or the
// device aliases of mapped pinned host memory) with SM loads and stores,
// then a system-scope fence. The Spark exchange stages and lands its rows
// this way: SM work stays in stream order behind stream memory waits.
int32_t cuteafd_sm_copy_regions(void* const* destinations, const void* const* sources,
    const uint64_t* bytes, uint32_t count, void* stream);

// One Spark rank's reply for `cuteafd_pull_frames`: the frames it arrived in
// (`count` of them, at most CUTEAFD_PULL_MAX_FRAMES), each `bytes` from
// `source` (a device-visible address of mapped pinned host memory: a
// registered receive slot) to `offset` bytes into the rank's plane. The host
// exchange worker writes it in mapped pinned memory before it publishes the
// rank's arrival; the kernel reads it when it runs, so a captured launch
// pulls whatever the worker published for that replay. A count of 0 copies
// nothing (the worker clears counts before it releases waits after a failure).
#define CUTEAFD_PULL_MAX_FRAMES 4
#define CUTEAFD_PULL_MAX_RANKS 8
typedef struct cuteafd_pull_entry {
  uint64_t source;
  uint64_t offset;
  uint64_t bytes;
} cuteafd_pull_entry_t;
typedef struct cuteafd_pull_frames {
  uint32_t count;
  uint32_t reserved;
  cuteafd_pull_entry_t frame[CUTEAFD_PULL_MAX_FRAMES];
} cuteafd_pull_frames_t;
// On the stream: pulls `count` (1..8) ranks' published frames
// (`replies[i]`, mapped) into device planes `planes[i]` (16-byte aligned,
// `capacity[i]` bytes). Frames 4-byte aligned in source, offset and size; a
// frame past its plane or misaligned copies nothing.
int32_t cuteafd_pull_frames(const cuteafd_pull_frames_t* const* replies, void* const* planes,
    const uint64_t* capacity, uint32_t count, void* stream);

// P2P probe for `cuteafd fabric --p2p`. Runs one measurement between devices
// `a` and `b` (peer access enabled both ways by the call) and writes the time
// per operation in microseconds (median of 5 repeats of `iterations` ops):
//   0 copy engine A->B one-way (cudaMemcpyAsync on B's stream)
//   1 SM pull A->B one-way (kernel on B reads A)
//   2 SM push A->B one-way (kernel on A writes B)
//   3 pinned host bounce A->host->B one-way (host-timed, includes a sync)
//   4 copy engine ping-pong with cross-device events (half round trip)
//   5 SM push ping-pong with cross-device events (half round trip)
//   6 SM push + flag ping-pong, eager launches (half round trip)
//   7 SM push + flag ping-pong, one CUDA graph per device (half round trip)
//   8 SM push + flag exchange (both push, both wait), eager (per round)
//   9 SM push + flag exchange, graphs (per round)
//  10 copy engine ping-pong with events, one multi-device graph (half round trip)
// `ingress` bit 0/1 keeps host->device DMA copies streaming into A/B during
// the measurement (a proxy for NIC->GPU ingress over the same PCIe links).
// `blocks` sizes the SM kernels (0: default). Returns a cudaError_t value.
int32_t cuteafd_p2p_probe(int32_t a, int32_t b, uint64_t bytes, int32_t test, uint32_t ingress,
    uint32_t iterations, uint32_t blocks, double* microseconds);
#ifdef __cplusplus
}
#endif
