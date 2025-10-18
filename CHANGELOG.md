# Changelog

## Unreleased

- Features:
  - GatewayStats: add `mark_rx`, `mark_tx`, `reset` and latency statistics (avg, p95, max).
  - Snapshot: `GatewayStatsSnapshot` dataclass with `.to_dict()`; snapshot() now includes `started_at`, `uptime_s`, and latency aggregates.
  - Rolling FPS windows configurable via `rolling_window_size` (default 20).
  - Latency buffer configurable via `latency_window_size` (default 200).

- Compatibility:
  - Backwards-compatible: existing `update_rx`, `update_tx`, `snapshot` and `set_last_rx_ts` remain available and unchanged in signature. New snapshot keys are additive.

- Tests:
  - Added unit tests for stats in `tests/service/gateway/test_stats.py`.

- Config:
  - `GatewayConfig` supports optional `stats` block with `rolling_window_size` and `latency_window_size`.
