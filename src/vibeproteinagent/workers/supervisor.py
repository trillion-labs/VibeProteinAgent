from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

from vibeproteinagent.adapters import InvalidOutputError, validate_output
from vibeproteinagent.core.errors import RETRYABLE_ERRORS, ErrorCode, NoGpuError, VPAError
from vibeproteinagent.core.log import Logger, get_file_logger
from vibeproteinagent.workers import ipc, paths
from vibeproteinagent.workers.admission import required_vram_mib
from vibeproteinagent.workers.worker import DEFAULT_MIN_FREE_VRAM_MIB, worker_main
from vibeproteinagent.workspace import (
    ArtifactStore,
    JobState,
    JobStore,
    RunStore,
    Workspace,
    now_us,
)

_POLL_INTERVAL_S = 0.25

# Once this many `high`-priority jobs have been dispatched back-to-back, the
# scheduler will pick the next queued ``normal | low`` job if any. Mirrors §7.
_ANTI_STARVATION_LIMIT = 8


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _WorkerHandle:
    """Bookkeeping for one spawned worker subprocess."""

    __slots__ = (
        "busy",
        "current_job_id",
        "gpu_id",
        "process",
        "socket_path",
    )

    def __init__(self, gpu_id: int, process: mp.Process) -> None:
        self.gpu_id = gpu_id
        self.process = process
        self.socket_path = paths.worker_socket(gpu_id)
        self.busy = False
        self.current_job_id: str | None = None


class Supervisor:
    """The single long-lived process that owns the job queue + worker pool."""

    def __init__(self, workspace_name: str, num_workers: int) -> None:
        self.workspace_name = workspace_name
        self.workspace = Workspace.resolve(workspace_name)
        if not self.workspace.db_path.exists():
            self.workspace = Workspace.create(workspace_name)
        self.num_workers = num_workers
        self.workers: list[_WorkerHandle] = []
        self._stop_event: asyncio.Event | None = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._draining = False
        self._server: asyncio.base_events.Server | None = None
        self._submit_locks: dict[int, asyncio.Lock] = {}
        # Cached per-worker state, refreshed by :meth:`_poll_worker_info_loop`.
        # Falls back to a synthesised snapshot if we have not yet polled the worker.
        self._worker_state: dict[int, dict[str, Any]] = {}
        # workflow_id -> gpu_id of the worker that ran the previous node, for
        # affinity-biased placement.
        self._workflow_affinity: dict[str, int] = {}
        self._consecutive_high_dispatched = 0
        self._registry: Any = None  # lazy
        # job_id -> last time we emitted a "waiting for GPUs" info log; throttles
        # the multi-GPU queue notice to once per 30 s per job (Codex RC4).
        self._gpu_wait_log_ts: dict[str, float] = {}
        # job_id -> tick count remaining during which we'll exclude a worker that
        # rejected with E_VRAM_EXHAUSTED (Codex RC5 self-healing requeue).
        self._vram_reject_blocklist: dict[str, dict[int, int]] = {}
        # job_id -> workers that have already rejected the exact params with
        # E_VRAM_EXHAUSTED. This is permanent for the job; once every eligible
        # worker has rejected, the job stays terminally failed instead of
        # cycling forever.
        self._vram_reject_gpus: dict[str, set[int]] = {}
        self._log: Logger = get_file_logger(
            "supervisor", paths.supervisor_log_file()
        )
        # Stream-frame subscribers (job_id -> list of writers). Used to forward
        # progress / retry events to live CLI followers. The on-disk
        # ``events.ndjson`` is always authoritative — subscribers are an
        # optional push channel.
        self._stream_subscribers: dict[str, list[asyncio.StreamWriter]] = {}
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        self.submit_timeout_s = float(os.environ.get("VPA_WORKER_RPC_TIMEOUT_S", "7200"))

    # --------------------------------------------------------------- start

    @classmethod
    def run_blocking(
        cls,
        workspace_name: str,
        num_workers: int,
        gpu_ids: list[int],
    ) -> None:
        """Entrypoint for the supervisor subprocess."""
        log_path = paths.supervisor_log_file()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = open(log_path, "a", buffering=1, encoding="utf-8")
        os.dup2(log_fp.fileno(), sys.stdout.fileno())
        os.dup2(log_fp.fileno(), sys.stderr.fileno())
        boot_log = get_file_logger("supervisor", log_path)
        boot_log.info(
            "starting",
            workspace=workspace_name, workers=num_workers, gpu_ids=str(gpu_ids),
        )
        supervisor = cls(workspace_name, num_workers)
        supervisor._gpu_ids = gpu_ids  # type: ignore[attr-defined]
        try:
            asyncio.run(supervisor._async_main(gpu_ids))
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            boot_log.error("fatal", error=str(exc))
            raise

    async def _async_main(self, gpu_ids: list[int]) -> None:
        self._stop_event = asyncio.Event()
        # Spawn workers.
        ctx = mp.get_context("spawn")
        spawn_started = time.monotonic()
        for gid in gpu_ids:
            proc = ctx.Process(
                target=worker_main,
                args=(gid, self.workspace_name),
                name=f"vpa-worker-{gid}",
                daemon=False,
            )
            proc.start()
            self._log.info("spawn_worker", gpu_id=gid, pid=proc.pid)
            handle = _WorkerHandle(gid, proc)
            self.workers.append(handle)
            self._submit_locks[gid] = asyncio.Lock()
            self._worker_state[gid] = self._synth_state(handle)
        # Wait for each worker socket to appear (torch init can be slow on N GPUs).
        await self._wait_workers_ready(timeout_s=45.0)
        self._log.info(
            "all_workers_ready",
            count=len(self.workers),
            t_ms=int((time.monotonic() - spawn_started) * 1000),
        )

        sock = paths.supervisor_socket()
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()
        self._server = await asyncio.start_unix_server(self._handle_client, path=str(sock))
        try:
            os.chmod(sock, 0o600)
        except OSError:
            pass
        pid_file = paths.supervisor_pid_file()
        pid_file.write_text(str(os.getpid()))
        self._log.info("listening", socket=str(sock), pid=os.getpid())

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: self._stop_event.set())  # type: ignore[union-attr]

        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        self._poll_task = asyncio.create_task(self._poll_worker_info_loop())

        try:
            await self._stop_event.wait()
        finally:
            self._log.info("shutdown_drain")
            self._draining = True
            for t in (self._scheduler_task, self._poll_task):
                if t is None:
                    continue
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            await self._drain_and_stop_workers()
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            with contextlib.suppress(FileNotFoundError):
                sock.unlink()
            with contextlib.suppress(FileNotFoundError):
                pid_file.unlink()
            self._log.info("stopped")

    async def _wait_workers_ready(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        ready: set[int] = set()
        while time.monotonic() < deadline and len(ready) < len(self.workers):
            for h in self.workers:
                if h.gpu_id in ready:
                    continue
                if h.socket_path.exists():
                    ready.add(h.gpu_id)
            if len(ready) == len(self.workers):
                break
            await asyncio.sleep(0.1)
        missing = [h.gpu_id for h in self.workers if h.gpu_id not in ready]
        if missing:
            self._log.warn("workers_not_ready", missing=str(missing))

    # --------------------------------------------------------------- registry

    def _ensure_registry(self) -> Any:
        if self._registry is None:
            from vibeproteinagent.adapters import ToolRegistry

            self._registry = ToolRegistry()
            self._registry.discover()
        return self._registry

    def _descriptor_for(self, name: str, version: str | None) -> Any | None:
        try:
            spec = f"=={version}" if version else "*"
            descriptor, _cls = self._ensure_registry().get(name, spec)
            return descriptor
        except VPAError:
            return None

    # --------------------------------------------------------------- worker_info polling

    def _synth_state(self, h: _WorkerHandle) -> dict[str, Any]:
        """Cheap stand-in cached entry used before the first poll succeeds."""
        return {
            "gpu_id": h.gpu_id,
            "pid": h.process.pid,
            "busy": h.busy,
            "current_job_id": h.current_job_id,
            "alive": h.process.is_alive(),
            "free_vram_mib": 0,
            "total_vram_mib": 0,
            "loaded_adapters": [],
        }

    async def _poll_worker_info_loop(self) -> None:
        """Refresh ``_worker_state`` from each worker every 250 ms."""
        while not self._draining:
            for h in self.workers:
                if self._draining:
                    break
                try:
                    info = await asyncio.wait_for(
                        self._request_worker_info(h), timeout=1.5
                    )
                    if info is not None:
                        info["alive"] = h.process.is_alive()
                        # Supervisor truth wins for busy / current_job_id
                        info["busy"] = h.busy
                        info["current_job_id"] = h.current_job_id
                        self._worker_state[h.gpu_id] = info
                except Exception:
                    # Connection refused while worker is restarting / busy is fine.
                    pass
            await asyncio.sleep(_POLL_INTERVAL_S)

    async def _request_worker_info(self, h: _WorkerHandle) -> dict[str, Any] | None:
        """Open a transient connection to ``h`` and request worker_info."""
        try:
            reader, writer = await asyncio.open_unix_connection(str(h.socket_path))
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        try:
            req = ipc.make_request({"op": "worker_info"})
            await ipc.send(writer, req)
            resp = await asyncio.wait_for(ipc.unpack(reader), timeout=1.5)
            if resp is None or resp.get("error"):
                return None
            return resp.get("body") or None
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    # --------------------------------------------------------------- scheduler

    async def _scheduler_loop(self) -> None:
        """Poll the DB for queued jobs; place them on the best worker per §7."""
        # Faster scheduler poll than the worker_info poll so noop-class jobs
        # don't sit idle in queue after their worker becomes available.
        scheduler_poll_s = 0.05
        while not self._draining:
            try:
                dispatches = self._place_one_pass()
                for handle, job_id, slaves in dispatches:
                    tasks = getattr(self, "_dispatch_tasks", None)
                    if tasks is None:
                        tasks = self._dispatch_tasks = set()
                    task = asyncio.create_task(
                        self._dispatch_to_worker(handle, job_id, slaves=slaves)
                    )
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            except Exception as exc:
                self._log.error("scheduler_error", error=str(exc))
            await asyncio.sleep(scheduler_poll_s)

    def _place_one_pass(self) -> list[tuple[_WorkerHandle, str, list[_WorkerHandle]]]:
        """One scheduler pass: try to dispatch as many queued jobs as we can.

        Returns a list of ``(handle, job_id, slaves)`` triples that the caller
        should fire-and-forget to the dispatch helper. Returned as a list (instead
        of being dispatched in-place) so unit tests can inspect placement decisions
        without an asyncio event loop.
        """
        decisions: list[tuple[_WorkerHandle, str, list[_WorkerHandle]]] = []
        # Tick down RC5 per-job worker exclusions before this pass.
        self._decay_vram_blocklist()
        store = JobStore(self.workspace)
        queued = store.list(state=JobState.QUEUED, limit=200, order="submitted_at_asc")
        if not queued:
            return decisions
        # Sort by (priority -> FIFO), matching JobStore.claim_next_queued semantics.
        prio_order = {"high": 0, "normal": 1, "low": 2}
        queued.sort(key=lambda r: (prio_order.get(r.priority.value, 99), r.submitted_at))

        # Anti-starvation: if too many `high` were dispatched in a row, find a
        # non-high job to dispatch first this pass.
        if self._consecutive_high_dispatched >= _ANTI_STARVATION_LIMIT:
            non_high_first = sorted(
                queued,
                key=lambda r: (0 if r.priority.value != "high" else 1,
                                prio_order.get(r.priority.value, 99),
                                r.submitted_at),
            )
            queued = non_high_first

        for rec in queued:
            if self._draining:
                return decisions
            # Drop jobs the user cancelled before they were claimed.
            if store.is_cancel_requested(rec.job_id):
                with contextlib.suppress(VPAError):
                    store.transition(
                        rec.job_id, JobState.CANCELLED,
                        error_code=ErrorCode.CANCELLED,
                        error_message="cancelled while queued",
                    )
                continue

            descriptor = self._descriptor_for(rec.tool_name, rec.tool_version)
            gpus_required = int(getattr(descriptor, "gpus_required", 1)) if descriptor else 1

            if self._fail_if_no_gpu_can_ever_run(rec, descriptor, store):
                continue

            if gpus_required > 1:
                placed = self._dispatch_multi_gpu(rec, descriptor, gpus_required,
                                                  store, decisions)
                if placed:
                    self._after_dispatch(rec)
                # Multi-GPU is exclusive; continue trying smaller jobs.
                continue

            chosen = self._pick_worker(rec, descriptor)
            if chosen is None:
                continue

            claimed = store.claim_next_queued(
                worker_id=chosen.gpu_id,
                gpu_ids=[chosen.gpu_id] if chosen.gpu_id >= 0 else [],
                job_id=rec.job_id,
            )
            if claimed is None:
                # Another tick (or a cancel) won the race for this job.
                continue
            chosen.busy = True
            chosen.current_job_id = claimed.job_id
            cached = self._worker_state.get(chosen.gpu_id)
            if cached is not None:
                cached["busy"] = True
                cached["current_job_id"] = claimed.job_id
            self._after_dispatch(claimed)
            decisions.append((chosen, claimed.job_id, []))
        return decisions

    def _after_dispatch(self, rec: Any) -> None:
        """Update the consecutive-high counter and workflow affinity record."""
        if getattr(rec, "priority", None) and rec.priority.value == "high":
            self._consecutive_high_dispatched += 1
        else:
            self._consecutive_high_dispatched = 0
        wf_id = getattr(rec, "workflow_id", None)
        # rec.gpu_ids may be empty for fresh records; the dispatch path will have
        # written worker_id via claim_next_queued.
        if wf_id and rec.gpu_ids:
            self._workflow_affinity[wf_id] = rec.gpu_ids[0]

    def _pick_worker(self, rec: Any, descriptor: Any | None) -> _WorkerHandle | None:
        """Score-based placement for a single job (see Architecture v1 §7)."""
        candidates = self._candidate_workers(rec, descriptor)
        # RC5: drop any worker that recently rejected this job for VRAM. The
        # blocklist decays after K=2 placement ticks (see _decay_vram_blocklist).
        blocklist = self._vram_reject_blocklist.get(rec.job_id) or {}
        if blocklist:
            candidates = [c for c in candidates if c.gpu_id not in blocklist]
        rejected = self._vram_reject_gpus.get(rec.job_id) or set()
        if rejected:
            candidates = [c for c in candidates if c.gpu_id not in rejected]
        if not candidates:
            return None

        # Affinity boost: if this workflow has run on a worker already, prefer it.
        wf_id = getattr(rec, "workflow_id", None)
        affinity_gpu = self._workflow_affinity.get(wf_id) if wf_id else None

        # Warm-pool boost: workers that already have this adapter loaded.
        warm_gpus = self._warm_workers_for(rec.tool_name, rec.tool_version)

        def score(h: _WorkerHandle) -> tuple[int, int, int, int]:
            cached = self._worker_state.get(h.gpu_id, {})
            free = int(cached.get("free_vram_mib", 0))
            # We want MAX free, so negate for ascending sort.
            return (
                0 if (affinity_gpu is not None and h.gpu_id == affinity_gpu) else 1,
                0 if h.gpu_id in warm_gpus else 1,
                -free,
                h.gpu_id,
            )

        candidates.sort(key=score)
        return candidates[0]

    def _candidate_workers(self, rec: Any, descriptor: Any | None) -> list[_WorkerHandle]:
        """Idle workers that satisfy GPU / VRAM requirements for ``rec``.

        VRAM is computed via :func:`workers.admission.required_vram_mib` so the
        scheduler agrees with the worker's pre-load check (Architecture v1 §7).
        """
        gpu_required = bool(getattr(descriptor, "gpu_required", False)) if descriptor else False
        # Scheduler-side: no live adapter, no bound params — descriptor minimum only.
        required_mib = required_vram_mib(descriptor) if descriptor is not None else (
            4 * 1024
        )
        warm_gpus = self._warm_workers_for(rec.tool_name, rec.tool_version)

        out: list[_WorkerHandle] = []
        for h in self.workers:
            if h.busy or not h.process.is_alive():
                continue
            has_gpu = h.gpu_id >= 0
            if gpu_required and not has_gpu:
                continue
            cached = self._worker_state.get(h.gpu_id, {})
            free = int(cached.get("free_vram_mib", 0))
            if h.gpu_id in warm_gpus:
                # Already loaded — only need a small headroom to actually run.
                headroom = DEFAULT_MIN_FREE_VRAM_MIB
                if gpu_required and free <= 0:
                    continue
                if free and free < headroom:
                    continue
            else:
                # Need to fit a fresh load.
                needed = max(required_mib, 4 * 1024) + DEFAULT_MIN_FREE_VRAM_MIB
                if gpu_required and free <= 0:
                    continue
                if free and free < needed:
                    continue
            out.append(h)
        return out

    def _fail_if_no_gpu_can_ever_run(
        self,
        rec: Any,
        descriptor: Any | None,
        store: JobStore,
    ) -> bool:
        """Terminally fail GPU-required jobs on CPU-only supervisors.

        Without this guard a single-GPU job can sit queued forever when the
        daemon only has the CPU worker (``gpu_id == -1``). Multi-GPU jobs have a
        similar permanent-failure guard in ``_dispatch_multi_gpu``; this covers
        the common ``gpus_required == 1`` path.
        """
        gpu_required = bool(getattr(descriptor, "gpu_required", False)) if descriptor else False
        if not gpu_required:
            return False
        if any(h.gpu_id >= 0 and h.process.is_alive() for h in self.workers):
            return False
        with contextlib.suppress(VPAError):
            store.transition(
                rec.job_id,
                JobState.FAILED,
                error_code=ErrorCode.NO_GPU,
                error_message=(
                    f"{rec.tool_name} requires a GPU worker; this daemon has only "
                    "CPU workers"
                ),
                error_details={
                    "tool": rec.tool_name,
                    "gpus_required": int(getattr(descriptor, "gpus_required", 1))
                    if descriptor
                    else 1,
                    "available_gpu_workers": 0,
                },
            )
        return True

    def _warm_workers_for(self, name: str, version: str | None) -> set[int]:
        gpus: set[int] = set()
        for gpu_id, state in self._worker_state.items():
            for entry in state.get("loaded_adapters", []) or []:
                if entry.get("name") == name and (
                    version is None or entry.get("version") == version
                ):
                    if entry.get("healthy") is False:
                        continue
                    gpus.add(gpu_id)
                    break
        return gpus

    def _dispatch_multi_gpu(
        self,
        rec: Any,
        descriptor: Any | None,
        gpus_required: int,
        store: JobStore,
        decisions: list[tuple[_WorkerHandle, str, list[_WorkerHandle]]],
    ) -> bool:
        """Multi-GPU placement admission (§7 / §15).

        VPA must not silently run a multi-GPU job on rank0 only. A descriptor
        must explicitly advertise ``distributed_entrypoint`` before the
        scheduler claims multiple workers. Adapters without that contract fail
        terminally instead of fake-dispatching.

        Codex round-2 RC4 distinguishes:

        * **Permanent failure**: ``gpus_required`` exceeds the total worker
          count. Marked ``failed`` with ``E_NO_GPU`` immediately — no amount
          of waiting will satisfy this job.
        * **Transient unavailability**: not enough idle/contiguous GPU
          workers right now. Returns ``False`` (job stays ``queued``); the
          scheduler re-evaluates next tick. A throttled info log fires
          every 30 s while the job continues to wait.
        """
        # All GPU workers (busy or not) — used to distinguish "never possible"
        # from "not possible right now".
        all_gpu_workers = sorted(
            (h for h in self.workers if h.gpu_id >= 0),
            key=lambda h: h.gpu_id,
        )
        if len(all_gpu_workers) < gpus_required:
            with contextlib.suppress(VPAError):
                store.transition(
                    rec.job_id, JobState.FAILED,
                    error_code=ErrorCode.NO_GPU,
                    error_message=(
                        f"{rec.tool_name} requires {gpus_required} GPU workers; "
                        f"this host has only {len(all_gpu_workers)}"
                    ),
                )
            return False

        distributed_entrypoint = getattr(descriptor, "distributed_entrypoint", None)
        if not distributed_entrypoint:
            with contextlib.suppress(VPAError):
                store.transition(
                    rec.job_id,
                    JobState.FAILED,
                    error_code=ErrorCode.NO_GPU,
                    error_message=(
                        f"{rec.tool_name} requests {gpus_required} GPUs, but this "
                        "adapter does not expose an end-to-end distributed worker "
                        "entrypoint yet"
                    ),
                    error_details={
                        "tool": rec.tool_name,
                        "gpus_required": gpus_required,
                        "available_gpu_workers": len(all_gpu_workers),
                        "reason": "multi_gpu_adapter_entrypoint_missing",
                    },
                )
            return False

        required_mib = required_vram_mib(descriptor) if descriptor is not None else 4 * 1024
        needed_mib = max(required_mib, 4 * 1024)
        gpu_workers = []
        for h in all_gpu_workers:
            if h.busy or not h.process.is_alive():
                continue
            cached = self._worker_state.get(h.gpu_id, {})
            free = int(cached.get("free_vram_mib", 0))
            needed_with_headroom = needed_mib + DEFAULT_MIN_FREE_VRAM_MIB
            if free <= 0:
                continue
            if free < needed_with_headroom:
                continue
            gpu_workers.append(h)
        chosen_block: list[_WorkerHandle] = []
        for i in range(len(gpu_workers) - gpus_required + 1):
            block = gpu_workers[i : i + gpus_required]
            block_ids = [h.gpu_id for h in block]
            if all(block_ids[j] + 1 == block_ids[j + 1] for j in range(len(block_ids) - 1)):
                chosen_block = block
                break
        if not chosen_block:
            self._maybe_log_waiting_for_gpus(rec.job_id, gpus_required)
            return False

        rank0 = chosen_block[0]
        claimed = store.claim_next_queued(
            worker_id=rank0.gpu_id,
            gpu_ids=[h.gpu_id for h in chosen_block],
            job_id=rec.job_id,
        )
        if claimed is None:
            return False
        rank0.busy = True
        rank0.current_job_id = claimed.job_id
        for h in chosen_block[1:]:
            h.busy = True
            h.current_job_id = claimed.job_id
        decisions.append((rank0, claimed.job_id, list(chosen_block[1:])))
        return True

    def _vram_reject_is_terminal(self, job_id: str, store: JobStore) -> bool:
        rec = store.get(job_id)
        if rec.error_code != ErrorCode.VRAM_EXHAUSTED:
            return False
        descriptor = self._descriptor_for(rec.tool_name, rec.tool_version)
        eligible = self._eligible_worker_ids_for_vram_retry(descriptor)
        rejected = self._vram_reject_gpus.get(job_id) or set()
        return bool(eligible and rejected.issuperset(eligible))

    def _handle_vram_reject(self, store: JobStore, job_id: str, gpu_id: int) -> None:
        """Self-heal a scheduler/worker VRAM disagreement (Codex round-2 RC5).

        The scheduler used the descriptor's declared minimum to admit this
        job onto ``gpu_id``, but once the worker actually loaded the
        adapter and ran its own ``estimate_vram(params)`` it concluded
        the job won't fit. The fix here is twofold:

        * Try each eligible worker at most once for this exact job/params.
        * If every eligible worker has rejected, keep the worker's terminal
          ``failed/E_VRAM_EXHAUSTED`` record instead of requeueing forever.
        * Otherwise re-queue without bumping ``retry_count`` and exclude the
          rejecting worker permanently for this job.
        """
        rec = store.get(job_id)
        rejected = self._vram_reject_gpus.setdefault(job_id, set())
        rejected.add(gpu_id)
        descriptor = self._descriptor_for(rec.tool_name, rec.tool_version)
        eligible = self._eligible_worker_ids_for_vram_retry(descriptor)
        if eligible and rejected.issuperset(eligible):
            self._log.warn(
                "vram_reject_terminal",
                job_id=job_id,
                rejected_gpus=str(sorted(rejected)),
                eligible_gpus=str(sorted(eligible)),
            )
            return
        try:
            store.requeue_after_worker_reject(job_id)
        except VPAError as exc:
            self._log.warn(
                "vram_requeue_failed", job_id=job_id, gpu_id=gpu_id, error=str(exc),
            )
            return
        self._vram_reject_blocklist.setdefault(job_id, {})[gpu_id] = 1 << 30
        self._log.info(
            "job_requeued_after_vram_reject", job_id=job_id, worker_id=gpu_id,
        )

    def _eligible_worker_ids_for_vram_retry(self, descriptor: Any | None) -> set[int]:
        gpu_required = bool(getattr(descriptor, "gpu_required", False)) if descriptor else False
        out: set[int] = set()
        for h in self.workers:
            if not h.process.is_alive():
                continue
            if gpu_required and h.gpu_id < 0:
                continue
            out.add(h.gpu_id)
        return out

    def _decay_vram_blocklist(self) -> None:
        """Tick down the per-job worker exclusion entries by one each pass."""
        empty_jobs: list[str] = []
        for job_id, by_gpu in list(self._vram_reject_blocklist.items()):
            for gpu_id in list(by_gpu.keys()):
                by_gpu[gpu_id] -= 1
                if by_gpu[gpu_id] <= 0:
                    by_gpu.pop(gpu_id, None)
            if not by_gpu:
                empty_jobs.append(job_id)
        for j in empty_jobs:
            self._vram_reject_blocklist.pop(j, None)

    def _maybe_log_waiting_for_gpus(self, job_id: str, gpus_required: int) -> None:
        """Emit ``job_waiting_for_gpus`` at most once per 30 s per job (RC4)."""
        last = self._gpu_wait_log_ts.get(job_id, 0.0)
        now = time.monotonic()
        if now - last < 30.0:
            return
        self._gpu_wait_log_ts[job_id] = now
        self._log.info(
            "job_waiting_for_gpus", job_id=job_id, g=gpus_required,
        )

    async def _dispatch_to_worker(
        self,
        handle: _WorkerHandle,
        job_id: str,
        *,
        slaves: list[_WorkerHandle] | None = None,
    ) -> None:
        """Send a submit request to the worker and await the terminal response.

        The worker writes progress events directly to
        ``<workspace>/jobs/<job_id>/events.ndjson`` (§10 NDJSON shape) while it
        runs the adapter. That keeps the supervisor's submit channel
        single-request-single-response under concurrent load (a multi-frame
        loop was tried and deadlocked when many connections were in flight).

        On success the worker returns ``body.completed=True`` plus a list of
        artifact descriptors (``[{field, path, kind, mime}, ...]``) and the
        finished output dict. The supervisor — the sole owner of the workspace
        SQLite handle — calls :class:`ArtifactStore.register` for each entry
        and only then transitions the job to ``completed`` (Codex review B1).

        If the worker socket closes mid-job (``resp is None``) the supervisor
        marks the in-flight job ``failed`` with ``E_INTERNAL``; the previous
        behaviour left it pinned to ``running`` forever (Codex review B2).
        """
        events_path = self.workspace.job_dir(job_id) / "events.ndjson"
        store = JobStore(self.workspace)
        artifact_store = ArtifactStore(self.workspace)
        run_store = RunStore(self.workspace)
        # Open a runs row up front (Codex review Re12). Best effort: any failure
        # to write the row must not break dispatch. Re10: persist the *full*
        # gpu_ids list for multi-GPU jobs (was rank0 only).
        run_id: str | None = None
        attempt = 1
        full_gpu_ids: list[int] = []
        if handle.gpu_id >= 0:
            full_gpu_ids = [handle.gpu_id]
            if slaves:
                full_gpu_ids += [s.gpu_id for s in slaves if s.gpu_id >= 0]
        try:
            rec_pre = store.get(job_id)
            attempt = int(rec_pre.retry_count) + 1
            run_id = run_store.start_run(
                job_id, attempt, worker_id=handle.gpu_id,
                gpu_ids=full_gpu_ids,
            )
        except Exception as exc:
            self._log.warn("run_record_failed", job_id=job_id, error=str(exc))

        self._log.info(
            "dispatch_job", job_id=job_id, worker_id=handle.gpu_id, gpu_id=handle.gpu_id,
        )
        dispatch_started = time.monotonic()
        try:
            if slaves:
                await self._dispatch_distributed_job(
                    store,
                    artifact_store,
                    job_id,
                    handle,
                    slaves,
                    full_gpu_ids,
                )
            else:
                resp = await self._submit_rpc(handle, {"op": "submit", "job_id": job_id})
                self._handle_submit_response(
                    store,
                    artifact_store,
                    job_id,
                    handle,
                    resp,
                    distributed=False,
                )
        except Exception as exc:
            self._log.error(
                "dispatch_failed",
                gpu_id=handle.gpu_id, job_id=job_id, error=str(exc),
                exception_type=type(exc).__name__,
            )
            # Re8: regardless of the underlying exception type
            # (ConnectionRefusedError, BrokenPipeError, asyncio.TimeoutError,
            # ...) the user-facing message is the contract. The exception
            # type lives in the structured log line above for diagnostics.
            self._fail_running_job(
                store, job_id,
                error_code=ErrorCode.INTERNAL,
                error_message="worker process died mid-job",
                details={"gpu_id": handle.gpu_id, "exception": str(exc)},
            )
        finally:
            handle.busy = False
            handle.current_job_id = None
            cached = self._worker_state.get(handle.gpu_id)
            if cached is not None:
                cached["busy"] = False
                cached["current_job_id"] = None
            if slaves:
                for s in slaves:
                    s.busy = False
                    s.current_job_id = None
                    cs = self._worker_state.get(s.gpu_id)
                    if cs is not None:
                        cs["busy"] = False
                        cs["current_job_id"] = None
            duration_ms = int((time.monotonic() - dispatch_started) * 1000)
            # Re11: log fidelity — distinct event names per terminal state.
            # The Phase 7 code logged ``job_completed`` even when the job
            # ended ``failed`` or ``cancelled``, which was loud and wrong.
            final_state: str | None = None
            final_error_code: str | None = None
            try:
                final_rec = store.get(job_id)
                final_state = final_rec.state.value
                final_error_code = final_rec.error_code
            except Exception:
                pass
            log_event = {
                "completed": "job_completed",
                "failed": "job_failed",
                "cancelled": "job_cancelled",
            }.get(final_state or "", "job_finished")
            log_kwargs: dict[str, Any] = {
                "job_id": job_id,
                "worker_id": handle.gpu_id,
                "duration_ms": duration_ms,
            }
            if log_event == "job_failed" and final_error_code:
                log_kwargs["error_code"] = final_error_code
            self._log.info(log_event, **log_kwargs)
            # Close the runs row with the job's terminal state (Re12).
            if run_id is not None:
                with contextlib.suppress(Exception):
                    if final_state is None:
                        final_state = store.get(job_id).state.value
                    vram_peak = None
                    cached = self._worker_state.get(handle.gpu_id, {})
                    used = cached.get("total_vram_mib", 0) - cached.get(
                        "free_vram_mib", 0
                    )
                    if used > 0:
                        vram_peak = int(used)
                    run_store.finish_run(
                        run_id,
                        exit_state=final_state,
                        duration_ms=duration_ms,
                        vram_peak_mib=vram_peak,
                    )
            # Auto-retry on retryable failures (Architecture v1 §5).
            with contextlib.suppress(Exception):
                self._maybe_auto_retry(job_id, events_path)

    async def _submit_rpc(
        self,
        handle: _WorkerHandle,
        body: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Send one submit-like request to one persistent worker."""
        async with self._submit_locks[handle.gpu_id]:
            reader, writer = await asyncio.open_unix_connection(str(handle.socket_path))
            try:
                await ipc.send(writer, ipc.make_request(body))
                return await asyncio.wait_for(
                    ipc.unpack(reader),
                    timeout=self.submit_timeout_s,
                )
            finally:
                with contextlib.suppress(Exception):
                    writer.close()
                # ``wait_closed`` has hung under concurrent UDS load on 3.12.

    async def _dispatch_distributed_job(
        self,
        store: JobStore,
        artifact_store: ArtifactStore,
        job_id: str,
        rank0: _WorkerHandle,
        slaves: list[_WorkerHandle],
        gpu_ids: list[int],
    ) -> None:
        """Run a distributed job by sending one RPC to every claimed rank."""
        rec_for_dispatch = store.get(job_id)
        descriptor = self._descriptor_for(
            rec_for_dispatch.tool_name,
            rec_for_dispatch.tool_version,
        )
        entrypoint = getattr(descriptor, "distributed_entrypoint", None)
        handles = [rank0, *slaves]
        world_size = len(handles)
        master_addr = "127.0.0.1"
        master_port = _pick_free_port()

        async def _rank_call(rank: int, h: _WorkerHandle) -> tuple[int, _WorkerHandle, Any]:
            body = {
                "op": "submit_distributed",
                "job_id": job_id,
                "distributed_entrypoint": entrypoint,
                "gpu_ids": gpu_ids,
                "rank": rank,
                "world_size": world_size,
                "master_addr": master_addr,
                "master_port": master_port,
                "local_world_size": world_size,
            }
            try:
                return rank, h, await self._submit_rpc(h, body)
            except Exception as exc:  # transport/runtime envelope failure
                return rank, h, exc

        results = await asyncio.gather(
            *(_rank_call(rank, h) for rank, h in enumerate(handles))
        )
        failures: list[tuple[int, _WorkerHandle, str, str, dict[str, Any]]] = []
        rank0_resp: dict[str, Any] | None = None
        for rank, h, resp in sorted(results, key=lambda item: item[0]):
            if isinstance(resp, Exception):
                self._log.error(
                    "distributed_worker_dispatch_failed",
                    gpu_id=h.gpu_id,
                    rank=rank,
                    job_id=job_id,
                    error=str(resp),
                    exception_type=type(resp).__name__,
                )
                failures.append(
                    (
                        rank,
                        h,
                        ErrorCode.INTERNAL,
                        "distributed worker process failed mid-job",
                        {"gpu_id": h.gpu_id, "rank": rank, "exception": str(resp)},
                    )
                )
                continue
            failure = self._submit_response_failure(job_id, h, resp, rank=rank)
            if failure is not None:
                failures.append(failure)
                continue
            if rank == 0:
                rank0_resp = resp

        if failures:
            rank, h, code, message, details = failures[0]
            self._log.error(
                "distributed_worker_returned_error",
                gpu_id=h.gpu_id,
                rank=rank,
                job_id=job_id,
                **{"error.code": code, "error.message": message},
            )
            self._fail_running_job(
                store,
                job_id,
                error_code=code,
                error_message=message,
                details={**details, "gpu_ids": gpu_ids},
            )
            return

        if rank0_resp is not None:
            body = rank0_resp.get("body") or {}
            if body.get("completed"):
                self._finalize_completed_job(store, artifact_store, job_id, body)
                return

        self._fail_running_job(
            store,
            job_id,
            error_code=ErrorCode.INTERNAL,
            error_message="distributed rank0 did not return a completed result",
            details={"gpu_ids": gpu_ids},
        )

    def _submit_response_failure(
        self,
        job_id: str,
        handle: _WorkerHandle,
        resp: dict[str, Any] | None,
        *,
        rank: int,
    ) -> tuple[int, _WorkerHandle, str, str, dict[str, Any]] | None:
        """Return normalized submit failure info without mutating job state."""
        if resp is None:
            return (
                rank,
                handle,
                ErrorCode.INTERNAL,
                "worker process died mid-job",
                {"gpu_id": handle.gpu_id, "rank": rank},
            )
        if resp.get("error"):
            err_body = resp["error"] or {}
            code = err_body.get("code") or ErrorCode.INTERNAL
            message = str(err_body.get("message") or "worker returned error")
            details = err_body.get("details") or {}
            return (
                rank,
                handle,
                code,
                message,
                {"gpu_id": handle.gpu_id, "rank": rank, **details},
            )
        body = resp.get("body") or {}
        error_code = body.get("error_code")
        if error_code:
            return (
                rank,
                handle,
                error_code,
                str(body.get("error_message") or "worker job failed"),
                body.get("error_details") or {"gpu_id": handle.gpu_id, "rank": rank},
            )
        if rank == 0 and not body.get("completed"):
            return (
                rank,
                handle,
                ErrorCode.INTERNAL,
                "distributed rank0 did not return a completed result",
                {"gpu_id": handle.gpu_id, "rank": rank, "body": body},
            )
        if rank != 0 and not body.get("rank_completed"):
            return (
                rank,
                handle,
                ErrorCode.INTERNAL,
                "distributed rank did not acknowledge completion",
                {"gpu_id": handle.gpu_id, "rank": rank, "body": body},
            )
        return None

    def _handle_submit_response(
        self,
        store: JobStore,
        artifact_store: ArtifactStore,
        job_id: str,
        handle: _WorkerHandle,
        resp: dict[str, Any] | None,
        *,
        distributed: bool,
        rank: int = 0,
    ) -> bool:
        """Apply a worker submit response. Returns True when it finalized success."""
        if resp is None:
            self._log.error(
                "worker_closed_mid_job",
                gpu_id=handle.gpu_id,
                rank=rank,
                job_id=job_id,
            )
            self._fail_running_job(
                store,
                job_id,
                error_code=ErrorCode.INTERNAL,
                error_message="worker process died mid-job",
                details={"gpu_id": handle.gpu_id, "rank": rank},
            )
            return False

        if resp.get("error"):
            err_body = resp["error"] or {}
            code = err_body.get("code") or ErrorCode.INTERNAL
            message = str(err_body.get("message") or "worker returned error")
            details = err_body.get("details") or {}
            self._log.error(
                "worker_returned_error",
                gpu_id=handle.gpu_id,
                rank=rank,
                job_id=job_id,
                **{"error.code": code, "error.message": message},
            )
            self._fail_running_job(
                store,
                job_id,
                error_code=code,
                error_message=message,
                details={"gpu_id": handle.gpu_id, "rank": rank, **details},
            )
            return False

        body = resp.get("body") or {}
        if body.get("completed"):
            if distributed and rank != 0:
                return False
            self._finalize_completed_job(store, artifact_store, job_id, body)
            return True

        error_code = body.get("error_code")
        if error_code:
            if not distributed and error_code == ErrorCode.VRAM_EXHAUSTED:
                self._handle_vram_reject(store, job_id, handle.gpu_id)
                return False
            self._fail_running_job(
                store,
                job_id,
                error_code=error_code,
                error_message=str(body.get("error_message") or "worker job failed"),
                details=body.get("error_details") or {"gpu_id": handle.gpu_id, "rank": rank},
            )
            return False

        if distributed and body.get("rank_completed"):
            return False

        self._fail_running_job(
            store,
            job_id,
            error_code=ErrorCode.INTERNAL,
            error_message="worker returned an incomplete submit response",
            details={"gpu_id": handle.gpu_id, "rank": rank, "body": body},
        )
        return False

    def _fail_running_job(
        self,
        store: JobStore,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Transition a ``running`` job to ``failed`` defensively.

        Used when the worker socket closes mid-job (B2), the dispatch raises,
        or anywhere else the supervisor observes the run has ended without a
        clean response. Idempotent — if the job has already reached a terminal
        state (race with the worker's own transition) we leave it alone.
        """
        try:
            rec = store.get(job_id)
        except VPAError:
            return
        if rec.state is not JobState.RUNNING:
            return
        with contextlib.suppress(VPAError):
            store.transition(
                rec.job_id, JobState.FAILED,
                error_code=error_code, error_message=error_message,
                error_details=details,
            )

    def _finalize_completed_job(
        self,
        store: JobStore,
        artifact_store: ArtifactStore,
        job_id: str,
        body: dict[str, Any],
    ) -> None:
        """Register the worker-reported artifacts then transition COMPLETED.

        ``body`` is the response body from the worker's success path: shape
        ``{output, artifacts: [{field, path, kind, mime}, ...], duration_ms}``.
        Per Architecture v1 §3 / §6 / §9, the supervisor is the only writer of
        the artifacts pool; the worker's job is to surface what it produced.

        Codex round-2 RC3: before persisting ``output`` we validate it
        against the descriptor's ``output_schema``. A schema violation
        becomes a normal terminal failure with ``E_INVALID_OUTPUT`` so a
        buggy adapter cannot silently store the wrong shape.
        """
        artifacts = body.get("artifacts") or []
        for art in artifacts:
            try:
                field = art["field"]
                path = art["path"]
            except (KeyError, TypeError):
                continue
            kind = art.get("kind") or "file"
            mime = art.get("mime")
            with contextlib.suppress(Exception):
                artifact_store.register(
                    job_id, field, path, kind=kind, mime=mime,
                )
        try:
            rec = store.get(job_id)
        except VPAError:
            return
        if rec.state is not JobState.RUNNING:
            return

        output = body.get("output")
        if isinstance(output, dict):
            try:
                descriptor = self._descriptor_for(rec.tool_name, rec.tool_version)
                if descriptor is not None:
                    validate_output(descriptor, output)
            except InvalidOutputError as exc:
                self._log.error(
                    "output_schema_violation",
                    job_id=job_id, tool=rec.tool_name,
                    field=exc.details.get("field"),
                )
                with contextlib.suppress(VPAError):
                    store.transition(
                        rec.job_id, JobState.FAILED,
                        error_code=exc.code, error_message=exc.message,
                        error_details=exc.details or None,
                    )
                return
            except Exception as exc:
                # Defensive: a bug in descriptor loading or output_validator
                # should not pin the job at RUNNING forever. Log and let the
                # COMPLETED transition below run.
                self._log.warn(
                    "output_validation_skipped",
                    job_id=job_id, tool=rec.tool_name, error=str(exc),
                )

        with contextlib.suppress(VPAError):
            store.transition(
                rec.job_id, JobState.COMPLETED,
                output=output,
            )

    def _maybe_auto_retry(self, job_id: str, events_path: Path) -> None:
        """Re-queue a failed job if its error_code is in :data:`RETRYABLE_ERRORS`."""
        store = JobStore(self.workspace)
        rec = store.get(job_id)
        if rec.state is not JobState.FAILED:
            return
        if rec.error_code not in RETRYABLE_ERRORS:
            return
        if rec.error_code == ErrorCode.VRAM_EXHAUSTED and self._vram_reject_is_terminal(
            job_id, store
        ):
            return
        if rec.retry_count >= rec.max_retries:
            return
        try:
            updated = store.requeue_failed(job_id)
        except VPAError:
            return
        # Emit a retrying event (NDJSON + subscribers).
        evt = {
            "type": "retrying",
            "job_id": job_id,
            "ts": time.time(),
            "payload": {
                "attempt": updated.retry_count,
                "reason": rec.error_code,
            },
        }
        with contextlib.suppress(Exception):
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evt) + "\n")
        # Best-effort fire-and-forget to subscribers (no-op if no loop running).
        body = {
            "event": "retrying",
            "job_id": job_id,
            "attempt": updated.retry_count,
            "reason": rec.error_code,
            "ts": int(time.time() * 1000),
        }
        subs = getattr(self, "_stream_subscribers", {}).get(job_id) or []
        for sub in list(subs):
            with contextlib.suppress(Exception):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    task = loop.create_task(ipc.send(sub, ipc.make_stream("", body)))
                    task.add_done_callback(lambda _task: None)

    # --------------------------------------------------------------- client requests

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                req = await ipc.unpack(reader)
                if req is None:
                    return
                resp = await self._dispatch(req)
                await ipc.send(writer, resp)
                if req.get("body", {}).get("op") == "shutdown":
                    self._stop_event.set()  # type: ignore[union-attr]
                    return
        except (ConnectionResetError, BrokenPipeError):
            return
        except Exception as exc:
            self._log.error("client_error", error=str(exc))
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        body = req.get("body") or {}
        op = body.get("op")
        mid = req.get("id", "")
        try:
            if op == "submit":
                return await self._op_submit(mid, body)
            if op == "cancel":
                return await self._op_cancel(mid, body)
            if op == "status":
                return await self._op_status(mid, body)
            if op == "warm":
                return await self._op_warm(mid, body)
            if op == "worker_info":
                return ipc.make_response(
                    mid, {"workers": [self._worker_info(w) for w in self.workers]}
                )
            if op == "shutdown":
                return ipc.make_response(mid, {"ok": True})
            return ipc.make_response(
                mid, {}, error=ipc.make_error(ErrorCode.INVALID_PARAMS, f"unknown op {op!r}"),
            )
        except VPAError as exc:
            return ipc.make_response(mid, {}, error=ipc.error_from_exception(exc))
        except Exception as exc:
            return ipc.make_response(mid, {}, error=ipc.error_from_exception(exc))

    def _worker_info(self, h: _WorkerHandle) -> dict[str, Any]:
        cached = self._worker_state.get(h.gpu_id, {})
        info = {
            "gpu_id": h.gpu_id,
            "pid": h.process.pid,
            "socket": str(h.socket_path),
            "busy": h.busy,
            "current_job_id": h.current_job_id,
            "alive": h.process.is_alive(),
            "free_vram_mib": int(cached.get("free_vram_mib", 0)),
            "total_vram_mib": int(cached.get("total_vram_mib", 0)),
            "loaded_adapters": list(cached.get("loaded_adapters", []) or []),
        }
        return info

    async def _op_submit(self, mid: str, body: dict[str, Any]) -> dict[str, Any]:
        """Accept a pre-queued job_id; scheduler will pick it up shortly."""
        job_id = body.get("job_id")
        if not isinstance(job_id, str):
            return ipc.make_response(
                mid, {},
                error=ipc.make_error(ErrorCode.INVALID_PARAMS, "submit requires job_id"),
            )
        store = JobStore(self.workspace)
        rec = store.get(job_id)
        return ipc.make_response(
            mid, {"job_id": rec.job_id, "state": rec.state.value, "queued": True},
        )

    async def _op_cancel(self, mid: str, body: dict[str, Any]) -> dict[str, Any]:
        job_id = body.get("job_id")
        if not isinstance(job_id, str):
            return ipc.make_response(
                mid, {},
                error=ipc.make_error(ErrorCode.INVALID_PARAMS, "cancel requires job_id"),
            )
        store = JobStore(self.workspace)
        rec = store.request_cancel(job_id)
        return ipc.make_response(mid, {"job_id": rec.job_id, "state": rec.state.value})

    async def _op_status(self, mid: str, body: dict[str, Any]) -> dict[str, Any]:
        job_id = body.get("job_id")
        if not isinstance(job_id, str):
            return ipc.make_response(
                mid, {},
                error=ipc.make_error(ErrorCode.INVALID_PARAMS, "status requires job_id"),
            )
        store = JobStore(self.workspace)
        rec = store.get(job_id)
        return ipc.make_response(
            mid,
            {
                "job_id": rec.job_id,
                "state": rec.state.value,
                "output": rec.output,
                "error_code": rec.error_code,
                "error_message": rec.error_message,
                "error_details": rec.error_details,
            },
        )

    async def _op_warm(self, mid: str, body: dict[str, Any]) -> dict[str, Any]:
        """Forward a warm request to a set of workers; aggregate per-worker results."""
        tool = body.get("tool")
        version_spec = body.get("version_spec") or "*"
        target_gpus = body.get("gpu_ids")
        if not isinstance(tool, str):
            return ipc.make_response(
                mid, {},
                error=ipc.make_error(ErrorCode.INVALID_PARAMS, "warm requires tool"),
            )
        if target_gpus is None:
            targets = list(self.workers)
        else:
            wanted = {int(g) for g in target_gpus}
            targets = [h for h in self.workers if h.gpu_id in wanted]

        # Fan out one warm RPC per target worker.
        async def _one(h: _WorkerHandle) -> dict[str, Any]:
            # Codex round-3 Re2: snapshot worker_state *before* we touch
            # the worker so the operator can tell waiting-on-busy apart
            # from cold-load. Truth source is the supervisor's view of
            # ``h.busy`` (kept in sync with claim_next_queued + dispatch
            # finalisation); the polled worker_info cache is only an
            # advisory mirror.
            state_before = "busy" if h.busy else "idle"
            t0 = time.monotonic()
            try:
                reader, writer = await asyncio.open_unix_connection(str(h.socket_path))
            except (FileNotFoundError, ConnectionRefusedError) as e:
                return {
                    "gpu_id": h.gpu_id,
                    "success": False,
                    "error": str(e),
                    "worker_state_before": state_before,
                    "waited_for_busy_worker_ms": 0,
                }
            try:
                req = ipc.make_request(
                    {"op": "warm", "tool": tool, "version_spec": version_spec}
                )
                await ipc.send(writer, req)
                # Cold-load on heavy adapters (Protenix ~110s, PXDesign ~95s)
                # exceeds the previous 60s budget. 240s gives ~2x headroom.
                resp = await asyncio.wait_for(ipc.unpack(reader), timeout=240.0)
                waited_ms = int((time.monotonic() - t0) * 1000)
                if resp is None:
                    return {
                        "gpu_id": h.gpu_id,
                        "success": False,
                        "error": "no response",
                        "worker_state_before": state_before,
                        "waited_for_busy_worker_ms": waited_ms,
                    }
                if resp.get("error"):
                    err = resp["error"]
                    return {
                        "gpu_id": h.gpu_id,
                        "success": False,
                        "error_code": err.get("code"),
                        "error": err.get("message"),
                        "worker_state_before": state_before,
                        "waited_for_busy_worker_ms": waited_ms,
                    }
                rb = resp.get("body") or {}
                return {
                    "gpu_id": h.gpu_id,
                    "success": True,
                    "vram_used_mib": rb.get("vram_used_mib", 0),
                    "free_vram_mib": rb.get("free_vram_mib", 0),
                    "pin_warm": rb.get("pin_warm", False),
                    "healthy": rb.get("healthy", True),
                    "warning": rb.get("warning"),
                    "worker_state_before": state_before,
                    "waited_for_busy_worker_ms": waited_ms,
                }
            finally:
                with contextlib.suppress(Exception):
                    writer.close()
                    await writer.wait_closed()

        results = await asyncio.gather(*[_one(h) for h in targets])
        # Refresh cached state opportunistically (next poll cycle will fully refresh).
        return ipc.make_response(
            mid,
            {"tool": tool, "version_spec": version_spec, "warmed": list(results)},
        )

    # --------------------------------------------------------------- shutdown

    async def _drain_and_stop_workers(self, *, timeout_s: float = 10.0) -> None:
        """Best-effort: ask workers to shutdown, then SIGTERM, then SIGKILL."""
        # Step 1: ask politely.
        for h in self.workers:
            with contextlib.suppress(Exception):
                reader, writer = await asyncio.open_unix_connection(str(h.socket_path))
                req = ipc.make_request({"op": "shutdown"})
                await ipc.send(writer, req)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(ipc.unpack(reader), timeout=1.0)
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
        # Step 2: SIGTERM + join.
        deadline = time.monotonic() + timeout_s
        for h in self.workers:
            if h.process.is_alive():
                with contextlib.suppress(Exception):
                    h.process.terminate()
        for h in self.workers:
            remaining = max(0.0, deadline - time.monotonic())
            with contextlib.suppress(Exception):
                h.process.join(timeout=remaining or 0.1)
        # Step 3: SIGKILL stragglers.
        for h in self.workers:
            if h.process.is_alive():
                with contextlib.suppress(Exception):
                    h.process.kill()
                    h.process.join(timeout=2.0)


def _stream_body_to_ndjson(body: dict[str, Any], job_id: str) -> dict[str, Any] | None:
    """Translate a worker stream-frame body into the §10 NDJSON event shape."""
    event = body.get("event")
    if not isinstance(event, str):
        return None
    # ts: worker sends integer ms; §10 NDJSON wants float seconds.
    raw_ts = body.get("ts")
    if isinstance(raw_ts, (int, float)):
        ts = float(raw_ts) / 1000.0 if isinstance(raw_ts, int) else float(raw_ts)
    else:
        ts = time.time()
    payload: dict[str, Any] = {}
    if event == "progress":
        payload["fraction"] = body.get("fraction")
    elif event == "retrying":
        payload["attempt"] = body.get("attempt")
        payload["reason"] = body.get("reason")
    else:
        # Pass through any other field that isn't a meta field.
        payload = {
            k: v for k, v in body.items()
            if k not in ("event", "job_id", "ts")
        }
    return {"type": event, "job_id": job_id, "ts": ts, "payload": payload}


def _pick_free_port() -> int:
    """Return a currently-unused TCP port (best-effort)."""
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return int(s.getsockname()[1])


def supervisor_main(workspace_name: str, num_workers: int, gpu_ids: list[int]) -> None:
    """Subprocess entry point (passed to multiprocessing.spawn)."""
    Supervisor.run_blocking(workspace_name, num_workers, gpu_ids)


def resolve_default_gpu_ids(num_workers: int | None = None) -> list[int]:
    """Return the gpu id list to spawn workers for.

    Without torch / CUDA, returns ``[-1]`` (single CPU-only worker).
    """
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.device_count()
            wanted = num_workers if num_workers is not None else min(total, 8)
            wanted = max(1, min(wanted, total))
            return list(range(wanted))
    except Exception:
        pass
    return [-1]


# Re-exports
NoGpuError = NoGpuError
now_us = now_us

__all__ = ["Supervisor", "resolve_default_gpu_ids", "supervisor_main"]
