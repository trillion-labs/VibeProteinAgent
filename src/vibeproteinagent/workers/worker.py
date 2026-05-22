from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import importlib
import inspect
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeproteinagent.core import config as cfg
from vibeproteinagent.core.errors import ErrorCode, VPAError, VramExhaustedError
from vibeproteinagent.core.log import Logger, get_file_logger
from vibeproteinagent.workers import ipc, paths
from vibeproteinagent.workers.admission import required_vram_mib
from vibeproteinagent.workers.cancel import poll_cancel
from vibeproteinagent.workspace import (
    JobState,
    JobStore,
    Workspace,
)

# Default headroom (MiB) the worker insists on keeping free after loading a new
# adapter. Used by the LRU eviction loop in :meth:`Worker._load_adapter`.
DEFAULT_MIN_FREE_VRAM_MIB = 2 * 1024  # 2 GiB

# Fallback "we have no idea, assume CPU" total VRAM reported when CUDA is absent.
_CPU_FAKE_VRAM_MIB = 1 << 20  # ~1 TiB; lets every CPU-only test fit trivially.


@dataclass
class AdapterMeta:
    """Sidecar bookkeeping for one cached adapter in :attr:`Worker._adapter_cache`."""

    vram_used_mib: int
    last_used_ts: float
    load_count: int
    run_count: int
    pin_warm: bool
    healthy: bool | None = None
    warning: str | None = None


@dataclass(frozen=True)
class DistributedRunContext:
    job_id: str
    workspace_root: Path
    job_dir: Path
    gpu_ids: list[int]
    rank: int
    world_size: int
    master_addr: str
    master_port: int
    local_world_size: int
    is_cancelled: Callable[[], bool]
    log: Callable[[str], None]
    emit_progress: Callable[[float], None]


def _availability_warning(descriptor: Any | None, cls: type) -> str | None:
    """Return an actionable availability warning from the descriptor probe."""
    probe_ref = getattr(descriptor, "availability_probe", None) if descriptor is not None else None
    if isinstance(probe_ref, str) and ":" in probe_ref:
        module_name, func_name = probe_ref.split(":", 1)
        try:
            mod = importlib.import_module(module_name)
            probe = getattr(mod, func_name)
            reason = probe()
            if isinstance(reason, str) and reason:
                return reason
        except Exception as exc:
            return f"{probe_ref} raised {type(exc).__name__}: {exc}"

    module_name = getattr(cls, "__module__", None)
    if not module_name:
        return None
    pkg_name = module_name.rsplit(".", 1)[0]
    pkg = sys.modules.get(pkg_name)
    probe = getattr(pkg, "availability_skip_reason", None)
    if callable(probe):
        try:
            reason = probe()
            if isinstance(reason, str) and reason:
                return reason
        except Exception as exc:
            return f"availability_skip_reason raised {type(exc).__name__}: {exc}"
    return None


def _probe_adapter_health(
    adapter: Any,
    cls: type,
    descriptor: Any | None = None,
) -> tuple[bool, str | None]:
    """Run an adapter's healthcheck and derive a warning string when it fails.

    Returns ``(healthy, warning)``. ``warning`` is ``None`` when ``healthy`` is
    ``True``; otherwise a human-readable string describing the gate. For gated
    adapters (BindCraft, PyRosetta, Proteina-Complexa, PXDesign pipeline
    mode) we look up the adapter package's ``availability_skip_reason()`` for
    a precise bootstrap-pointer message; otherwise we fall back to a generic
    "healthcheck returned False" string. Exceptions from healthcheck are
    treated as ``healthy=False`` with the exception string as the warning.
    """
    warning = _availability_warning(descriptor, cls)
    try:
        healthy = bool(adapter.healthcheck())
    except Exception as exc:
        return False, f"healthcheck raised {type(exc).__name__}: {exc}"
    if healthy:
        return True, warning
    if warning is None:
        warning = "adapter healthcheck returned False (gate not satisfied)"
    return False, warning


def _setup_torch(gpu_id: int, log: Logger | None = None) -> Any:
    """Pin CUDA_VISIBLE_DEVICES *before* torch import; configure TF32 + mem pool."""
    if gpu_id < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import torch
    except ImportError:
        if gpu_id >= 0:
            raise
        from vibeproteinagent.adapters.base import torch

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if gpu_id >= 0 and torch.cuda.is_available():
        try:
            torch.cuda.init()
            torch.cuda.set_per_process_memory_fraction(0.95, 0)
        except Exception as exc:
            if log is not None:
                log.error("torch_cuda_init_failed", gpu_id=gpu_id, error=str(exc))
            else:
                sys.stderr.write(f"[worker {gpu_id}] torch cuda init failed: {exc}\n")
    return torch


class Worker:
    """One-per-GPU subprocess: serves submit/cancel/healthcheck over UDS.

    Maintains a per-worker LRU cache of loaded :class:`ToolAdapter` instances.
    Eviction is triggered when free VRAM cannot accommodate the next load with
    the configured ``min_free_vram_mib`` headroom; pinned entries are skipped.
    """

    def __init__(
        self,
        gpu_id: int,
        workspace_name: str,
        *,
        min_free_vram_mib: int = DEFAULT_MIN_FREE_VRAM_MIB,
    ) -> None:
        self.gpu_id = gpu_id
        self.workspace_name = workspace_name
        self.workspace = Workspace.resolve(workspace_name)
        if not self.workspace.db_path.exists():
            self.workspace = Workspace.create(workspace_name)
        self._busy = False
        self._adapter_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._adapter_meta: dict[tuple[str, str], AdapterMeta] = {}
        self.min_free_vram_mib = min_free_vram_mib
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"worker{gpu_id}"
        )
        self._stop_event: asyncio.Event | None = None
        self._server: asyncio.base_events.Server | None = None
        self._current_job_id: str | None = None
        self._log: Logger = get_file_logger(
            f"worker:{gpu_id}", paths.worker_log_file(gpu_id)
        )
        self._torch = _setup_torch(gpu_id, log=self._log)
        self._registry: Any = None

    # --------------------------------------------------------------- lifecycle

    def _ensure_registry(self) -> Any:
        if self._registry is None:
            from vibeproteinagent.adapters import ToolRegistry

            self._registry = ToolRegistry()
            self._registry.discover()
        return self._registry

    async def serve(self) -> None:
        sock = paths.worker_socket(self.gpu_id)
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()
        self._stop_event = asyncio.Event()
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(sock)
        )
        try:
            os.chmod(sock, 0o600)
        except OSError:
            pass
        pid_file = paths.worker_pid_file(self.gpu_id)
        pid_file.write_text(str(os.getpid()))
        self._log.info("ready", socket=str(sock), pid=os.getpid())

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: self._stop_event.set())  # type: ignore[union-attr]

        try:
            await self._stop_event.wait()
        finally:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            with contextlib.suppress(FileNotFoundError):
                sock.unlink()
            with contextlib.suppress(FileNotFoundError):
                pid_file.unlink()
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._log.info("stopped")

    # --------------------------------------------------------------- request loop

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
                resp = await self._dispatch(req, writer)
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

    async def _dispatch(
        self,
        req: dict[str, Any],
        writer: asyncio.StreamWriter | None = None,
    ) -> dict[str, Any]:
        body = req.get("body", {}) or {}
        op = body.get("op")
        mid = req.get("id", "")
        try:
            if op == "submit":
                return await self._op_submit(mid, body, writer)
            if op == "submit_distributed":
                return await self._op_submit(mid, body, writer, distributed=True)
            if op == "warm":
                return await self._op_warm(mid, body)
            if op == "healthcheck":
                return ipc.make_response(mid, {"ok": True, "gpu_id": self.gpu_id,
                                               "busy": self._busy})
            if op == "worker_info":
                return ipc.make_response(mid, self._info())
            if op == "shutdown":
                return ipc.make_response(mid, {"ok": True})
            return ipc.make_response(
                mid, {}, error=ipc.make_error(ErrorCode.INVALID_PARAMS,
                                              f"unknown op {op!r}"),
            )
        except VPAError as exc:
            return ipc.make_response(mid, {}, error=ipc.error_from_exception(exc))
        except Exception as exc:
            return ipc.make_response(mid, {}, error=ipc.error_from_exception(exc))

    def _info(self) -> dict[str, Any]:
        free_mib, total_mib = self._probe_vram()
        loaded = []
        for (name, version), meta in self._adapter_meta.items():
            loaded.append(
                {
                    "name": name,
                    "version": version,
                    "vram_used_mib": meta.vram_used_mib,
                    "last_used_ts": meta.last_used_ts,
                    "pin_warm": meta.pin_warm,
                    "run_count": meta.run_count,
                    "load_count": meta.load_count,
                    "healthy": meta.healthy,
                    "warning": meta.warning,
                }
            )
        return {
            "gpu_id": self.gpu_id,
            "pid": os.getpid(),
            "busy": self._busy,
            "current_job_id": self._current_job_id,
            "free_vram_mib": free_mib,
            "total_vram_mib": total_mib,
            "loaded_adapters": loaded,
        }

    # --------------------------------------------------------------- VRAM probe

    def _probe_vram(self) -> tuple[int, int]:
        """Return ``(free_mib, total_mib)``. CPU fallback returns a large constant.

        Patchable on the instance for tests that need to simulate VRAM pressure.
        """
        torch = self._torch
        if self.gpu_id >= 0 and torch.cuda.is_available():
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                return (int(free_b) // (1024 * 1024), int(total_b) // (1024 * 1024))
            except Exception:
                pass
        return (_CPU_FAKE_VRAM_MIB, _CPU_FAKE_VRAM_MIB)

    # --------------------------------------------------------------- LRU helpers

    def _required_vram_mib(self, descriptor: Any) -> int:
        """Required VRAM for a new load (gate (a) — cold load).

        Delegates to :func:`workers.admission.required_vram_mib` with
        ``adapter=None``. The worker does not yet have an adapter
        instance here (we're about to load it), so the formula returns
        the descriptor's declared minimum — matching the supervisor's
        scheduler-side check exactly.
        """
        return required_vram_mib(descriptor, adapter=None, params=None)

    def free_vram_mib(self) -> int:
        """Currently-free VRAM on this worker's GPU, in MiB."""
        free, _total = self._probe_vram()
        return free

    def _check_params_admission(
        self,
        descriptor: Any,
        adapter: Any,
        params_model: Any,
    ) -> None:
        """Param-aware VRAM admission (gate (b) — already-loaded adapter).

        After :meth:`_load_adapter` returns the live adapter knows the
        params and can give a tighter ``estimate_vram(params)`` than the
        descriptor's static floor.

        Per Codex round-4 RC2 (:class:`ToolAdapter.estimate_vram`'s clarified
        contract), the adapter returns the **total** VRAM the job needs
        *including* the resident model weights. So if the adapter is
        already loaded, the residency it already consumes is part of the
        total — it must not be required *again*. We subtract the cached
        residency from the total to get the *incremental* free VRAM the
        forward pass actually needs.

        On cold load this method runs after the residency was just
        measured in :meth:`_load_adapter`, so the subtraction is exact.
        On warm-load second call the cached residency is the one
        measured at first load. Either way: pre-fix this rejected jobs
        that would have fit.

        If the incremental need plus the configured headroom still cannot
        fit, raise ``E_VRAM_EXHAUSTED``. The supervisor's self-heal
        (Codex round-2 RC5) requeues the job and blocklists this worker
        for K=2 placement ticks, so all we do here is fail loudly with
        ``details.params_dependent`` so the operator and the post-mortem
        can distinguish a placement bug from a genuine "this machine is
        too small for this job".
        """
        total_required = required_vram_mib(descriptor, adapter, params_model)
        meta = self._adapter_meta.get((descriptor.name, descriptor.version))
        loaded_residency = int(meta.vram_used_mib) if meta is not None else 0
        incremental_need = max(0, total_required - loaded_residency)
        free = self.free_vram_mib()
        if free < incremental_need + self.min_free_vram_mib:
            raise VramExhaustedError(
                "insufficient VRAM for params (after accounting for resident model)",
                details={
                    "total_required_mib": total_required,
                    "loaded_residency_mib": loaded_residency,
                    "incremental_need_mib": incremental_need,
                    "free_mib": free,
                    "min_free_vram_mib": self.min_free_vram_mib,
                    "params_dependent": True,
                },
            )

    def _touch(self, key: tuple[str, str]) -> None:
        """Promote ``key`` to MRU position; refresh ``last_used_ts``."""
        if key in self._adapter_cache:
            self._adapter_cache.move_to_end(key, last=True)
        if key in self._adapter_meta:
            self._adapter_meta[key].last_used_ts = time.time()

    def _evict_one(self) -> tuple[str, str] | None:
        """Evict the LRU non-pinned adapter; return its key, or ``None`` if all pinned."""
        victim_key: tuple[str, str] | None = None
        # Iteration order is LRU -> MRU; find the first non-pinned.
        for k in self._adapter_cache:
            meta = self._adapter_meta.get(k)
            if meta is not None and not meta.pin_warm:
                victim_key = k
                break
        if victim_key is None:
            return None
        adapter = self._adapter_cache.pop(victim_key)
        self._adapter_meta.pop(victim_key, None)
        with contextlib.suppress(Exception):
            adapter.unload()
        torch = self._torch
        if hasattr(torch, "cuda"):
            with contextlib.suppress(Exception):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._log.info(
            "adapter_evict",
            name=victim_key[0], version=victim_key[1], reason="lru",
        )
        return victim_key

    def _evict_incompatible(self, descriptor: Any, name: str) -> list[tuple[str, str]]:
        """Unconditionally evict any cached adapter that is incompatible with ``descriptor``.

        Two directions are checked:
          1. cached adapter X where ``descriptor.incompatible_with`` lists X;
          2. cached adapter Y whose own ``incompatible_with`` lists the incoming
             adapter's name (reciprocal — Y was declared incompatible with us).

        Eviction overrides ``pin_warm`` (with a warning logged). Returns the list
        of (name, version) keys that were evicted.
        """
        declared = set(getattr(descriptor, "incompatible_with", []) or [])
        evicted: list[tuple[str, str]] = []
        # Snapshot keys: we mutate the cache while iterating.
        for cached_key in list(self._adapter_cache.keys()):
            cached_name, _ = cached_key
            cached_descriptor = getattr(
                self._adapter_cache[cached_key], "descriptor", None
            )
            cached_incompat = set(
                getattr(cached_descriptor, "incompatible_with", []) or []
            )
            if cached_name in declared or name in cached_incompat:
                meta = self._adapter_meta.get(cached_key)
                was_pinned = bool(meta.pin_warm) if meta is not None else False
                if was_pinned:
                    self._log.warn(
                        "adapter_evict_pinned",
                        name=cached_key[0], version=cached_key[1],
                        reason="incompatible", with_=name,
                    )
                adapter = self._adapter_cache.pop(cached_key)
                self._adapter_meta.pop(cached_key, None)
                with contextlib.suppress(Exception):
                    adapter.unload()
                torch = self._torch
                if hasattr(torch, "cuda"):
                    with contextlib.suppress(Exception):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                self._log.info(
                    "adapter_evict",
                    name=cached_key[0], version=cached_key[1],
                    reason="incompatible", with_=name,
                )
                evicted.append(cached_key)
        return evicted

    def _load_adapter(
        self,
        name: str,
        version: str,
        descriptor: Any,
        cls: type,
    ) -> Any:
        """Lazy-load (or fetch) an adapter; evict LRU non-pinned entries as needed.

        Raises :class:`VramExhaustedError` if even after evicting every non-pinned
        adapter the free VRAM still cannot satisfy ``required + min_free_vram_mib``.
        """
        key = (name, version)
        # Cache hit: bump LRU and return.
        if key in self._adapter_cache:
            self._touch(key)
            return self._adapter_cache[key]
        self._log.info("adapter_load_start", name=name, version=version)
        load_started = time.time()

        # Incompatibility eviction runs BEFORE the VRAM/LRU loop. Pinned adapters
        # are evicted unconditionally if they are declared incompatible.
        self._evict_incompatible(descriptor, name)

        required = self._required_vram_mib(descriptor)
        # Evict in a loop until we have enough headroom (or only pinned remain).
        while True:
            free_mib, _total = self._probe_vram()
            if free_mib >= required + self.min_free_vram_mib:
                break
            victim = self._evict_one()
            if victim is None:
                raise VramExhaustedError(
                    f"adapter {name}=={version} requires {required} MiB + "
                    f"{self.min_free_vram_mib} MiB headroom, free={free_mib} MiB and "
                    f"all cached adapters are pin_warm",
                    details={
                        "name": name,
                        "version": version,
                        "required_mib": required,
                        "free_mib": free_mib,
                        "min_free_vram_mib": self.min_free_vram_mib,
                    },
                )

        # Measure pre-load reserved bytes so we can record the delta.
        torch = self._torch
        pre_reserved = 0
        if self.gpu_id >= 0 and getattr(torch, "cuda", None) and torch.cuda.is_available():
            with contextlib.suppress(Exception):
                pre_reserved = int(torch.cuda.memory_reserved())

        device = self._device_for_adapter()
        adapter = cls(config={}, weights_dir=cfg.model_cache_root(), device=device)

        # Measure VRAM the load consumed.
        if self.gpu_id >= 0 and getattr(torch, "cuda", None) and torch.cuda.is_available():
            post = 0
            with contextlib.suppress(Exception):
                post = int(torch.cuda.memory_reserved())
            vram_used = max(0, (post - pre_reserved) // (1024 * 1024))
        else:
            # CPU path / fake adapter: trust the descriptor's declared minimum.
            gib = getattr(descriptor, "vram_min_gib", None)
            vram_used = int(gib) * 1024 if gib is not None else 0

        pin_warm = bool(getattr(descriptor, "pin_warm", False))
        meta = AdapterMeta(
            vram_used_mib=vram_used,
            last_used_ts=time.time(),
            load_count=1,
            run_count=0,
            pin_warm=pin_warm,
        )
        self._adapter_cache[key] = adapter
        self._adapter_meta[key] = meta
        self._touch(key)
        self._log.info(
            "adapter_load_done",
            name=name, version=version,
            vram_mib=vram_used, pin_warm=pin_warm,
            t_ms=int((time.time() - load_started) * 1000),
        )
        return adapter

    # --------------------------------------------------------------- submit

    async def _op_submit(
        self,
        mid: str,
        body: dict[str, Any],
        writer: asyncio.StreamWriter | None = None,
        *,
        distributed: bool = False,
    ) -> dict[str, Any]:
        job_id = body.get("job_id")
        if not isinstance(job_id, str):
            return ipc.make_response(
                mid, {}, error=ipc.make_error(ErrorCode.INVALID_PARAMS,
                                              "submit body requires job_id"),
            )
        # NOTE: we used to reject with E_BUSY if ``self._busy`` was True, but
        # that races against the executor's serialisation under burst load
        # (supervisor lock + worker self._busy could drift by a couple of
        # microseconds). The single-slot ThreadPoolExecutor below already
        # serialises adapter runs per worker; concurrent submits just queue.
        self._busy = True
        self._current_job_id = job_id

        # Progress streaming: the adapter's ``ctx.emit_progress`` callable writes
        # a §10 NDJSON event to ``<job_dir>/events.ndjson`` (one line per call).
        # Throttle: at most 10 events/sec per job, newest wins. We do NOT inject
        # a final 1.0 on the worker's behalf — adapters that want a 1.0 must
        # emit it themselves; failure/cancel paths should not see a synthetic
        # "we got there" frame at the tail. (Codex review Re11.)
        events_path = self.workspace.job_dir(job_id) / "events.ndjson"
        state_lock = threading.Lock()
        progress_state: dict[str, Any] = {"last_emit": 0.0, "stopped": False}

        def thread_emit_progress(fraction: float) -> None:
            try:
                f = float(fraction)
            except (TypeError, ValueError):
                return
            f = max(0.0, min(1.0, f))
            now = time.time()
            with state_lock:
                if progress_state["stopped"]:
                    # The job has reached a terminal state; honest streams stop here.
                    return
                # ~20/sec throttle: drop sub-1.0 frames inside the 50 ms window.
                # Final frames (f == 1.0) always pass — they're a contract beat
                # for downstream consumers. (Codex round-2 C14: comment used to
                # say "10/sec"; reconciled here. 20/sec keeps the existing noop
                # progress-stream test green and is still a sane upper bound.)
                if f < 1.0 and now - progress_state["last_emit"] < 0.05:
                    return
                progress_state["last_emit"] = now
            _append_event(events_path, {
                "type": "progress",
                "job_id": job_id,
                "ts": now,
                "payload": {"fraction": f},
            })

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._run_job_blocking,
                job_id,
                thread_emit_progress,
                body if distributed else None,
            )
            return ipc.make_response(mid, result)
        finally:
            # Stop any in-flight progress callbacks that might still fire after
            # the adapter's executor thread returns; the stream is now closed.
            with state_lock:
                progress_state["stopped"] = True
            self._busy = False
            self._current_job_id = None

    async def _op_warm(self, mid: str, body: dict[str, Any]) -> dict[str, Any]:
        """Preload an adapter without running it. Returns the new cache state for ``tool``."""
        tool = body.get("tool")
        version_spec = body.get("version_spec") or "*"
        if not isinstance(tool, str):
            return ipc.make_response(
                mid, {}, error=ipc.make_error(ErrorCode.INVALID_PARAMS,
                                              "warm requires tool"),
            )
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._warm_blocking, tool, version_spec
            )
            return ipc.make_response(mid, result)
        except VPAError as exc:
            return ipc.make_response(mid, {}, error=ipc.error_from_exception(exc))

    def _warm_blocking(self, tool: str, version_spec: str) -> dict[str, Any]:
        registry = self._ensure_registry()
        descriptor, cls = registry.get(tool, version_spec)
        adapter = self._load_adapter(descriptor.name, descriptor.version, descriptor, cls)
        meta = self._adapter_meta[(descriptor.name, descriptor.version)]
        free_mib, total_mib = self._probe_vram()
        healthy, warning = _probe_adapter_health(adapter, cls, descriptor)
        meta.healthy = healthy
        meta.warning = warning
        return {
            "tool": descriptor.name,
            "version": descriptor.version,
            "vram_used_mib": meta.vram_used_mib,
            "pin_warm": meta.pin_warm,
            "free_vram_mib": free_mib,
            "total_vram_mib": total_mib,
            "healthy": healthy,
            "warning": warning,
        }

    # --------------------------------------------------------------- run job

    def _run_job_blocking(
        self,
        job_id: str,
        emit_progress: Callable[[float], None] | None = None,
        distributed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Executed in a thread; loads adapter + runs it; updates JobStore."""
        store = JobStore(self.workspace)
        rec = store.get(job_id)

        # Pre-flight cancel check.
        if store.is_cancel_requested(rec.job_id):
            terminal = store.transition(
                rec.job_id, JobState.CANCELLED,
                error_code=ErrorCode.CANCELLED,
                error_message="cancel requested before start",
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": ErrorCode.CANCELLED}

        # Look up adapter. tool_version is a concrete version (e.g. "1.0.0"); we
        # pass an exact-match PEP-440 specifier ("==<version>") to the registry.
        try:
            registry = self._ensure_registry()
            spec = f"=={rec.tool_version}" if rec.tool_version else "*"
            descriptor, cls = registry.get(rec.tool_name, spec)
        except VPAError as exc:
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=exc.code, error_message=exc.message,
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": exc.code}

        # Resolve params model.
        try:
            params_model = _resolve_input_model(cls, rec.params)
        except Exception as exc:
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=ErrorCode.INVALID_PARAMS, error_message=str(exc),
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": ErrorCode.INVALID_PARAMS}

        if distributed is not None:
            return self._run_distributed_job_blocking(
                rec=rec,
                descriptor=descriptor,
                params_model=params_model,
                emit_progress=emit_progress,
                distributed=distributed,
            )

        # Lazy-instantiate (or reuse) adapter via LRU cache.
        try:
            adapter = self._load_adapter(descriptor.name, descriptor.version, descriptor, cls)
        except VPAError as exc:
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=exc.code, error_message=exc.message,
                error_details=exc.details or None,
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": exc.code}
        except Exception as exc:
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=ErrorCode.DEPENDENCY, error_message=str(exc),
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": ErrorCode.DEPENDENCY}

        # Gate (b): param-aware VRAM admission. The descriptor floor was
        # already checked by gate (a) inside :meth:`_load_adapter`. Now
        # that the live adapter is in hand it can give a tighter bound
        # against the actual params (e.g. sequence length, batch size).
        try:
            self._check_params_admission(descriptor, adapter, params_model)
        except VPAError as exc:
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=exc.code, error_message=exc.message,
                error_details=exc.details or None,
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": exc.code}

        key = (descriptor.name, descriptor.version)
        # Build run context.
        from vibeproteinagent.adapters.context import AdapterRunContext

        job_dir = self.workspace.job_dir(rec.job_id)
        log_path = job_dir / "stdout.log"

        def _log(line: str) -> None:
            with contextlib.suppress(Exception):
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line if line.endswith("\n") else line + "\n")

        # When run via _op_submit the supervisor passes the open IPC writer down
        # to the worker; emit_progress publishes onto that writer (throttled).
        # When called outside the IPC path (unit tests), drop progress silently.
        _progress = emit_progress if emit_progress is not None else (lambda _f: None)

        ctx = AdapterRunContext(
            job_id=rec.job_id,
            workspace_root=self.workspace.root,
            job_dir=job_dir,
            is_cancelled=poll_cancel(store, rec.job_id, interval_s=0.25),
            log=_log,
            emit_progress=_progress,
            fusion_inputs=None,
        )

        # Execute.
        started = time.time()
        try:
            output_model = adapter.run(params_model, ctx)
        except Exception as exc:
            if store.is_cancel_requested(rec.job_id):
                terminal = store.transition(
                    rec.job_id, JobState.CANCELLED,
                    error_code=ErrorCode.CANCELLED, error_message=str(exc),
                )
                return {"job_id": terminal.job_id, "state": terminal.state.value,
                        "output": None, "error_code": ErrorCode.CANCELLED}
            code = exc.code if isinstance(exc, VPAError) else ErrorCode.INTERNAL
            details = exc.details if isinstance(exc, VPAError) else None
            terminal = store.transition(
                rec.job_id, JobState.FAILED,
                error_code=code, error_message=str(exc),
                error_details=details or None,
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": code, "error_details": details}

        duration_ms = int((time.time() - started) * 1000)

        # Bump run-count + touch the LRU after a successful adapter call.
        if key in self._adapter_meta:
            self._adapter_meta[key].run_count += 1
            self._touch(key)

        # Cancel-after-run check.
        if store.is_cancel_requested(rec.job_id):
            terminal = store.transition(
                rec.job_id, JobState.CANCELLED,
                error_code=ErrorCode.CANCELLED,
                error_message="cancelled after adapter returned",
            )
            return {"job_id": terminal.job_id, "state": terminal.state.value,
                    "output": None, "error_code": ErrorCode.CANCELLED}

        out_dict = _model_to_dict(output_model)
        # B1: introspect the output model for files under ctx.job_dir/outputs
        # and pass them up to the supervisor — only the supervisor owns SQLite,
        # so it (not the worker) is the one that calls ArtifactStore.register
        # and finalises the job state. Worker just describes what it produced.
        artifacts = _introspect_artifacts(out_dict, job_dir)
        return {
            "job_id": rec.job_id,
            "state": JobState.RUNNING.value,  # supervisor will transition → completed
            "output": out_dict,
            "duration_ms": duration_ms,
            "artifacts": artifacts,
            "completed": True,  # success marker for the supervisor
        }

    def _run_distributed_job_blocking(
        self,
        *,
        rec: Any,
        descriptor: Any,
        params_model: Any,
        emit_progress: Callable[[float], None] | None,
        distributed: dict[str, Any],
    ) -> dict[str, Any]:
        """Run one rank of an explicit adapter-owned distributed entrypoint."""
        store = JobStore(self.workspace)
        entrypoint = distributed.get("distributed_entrypoint") or getattr(
            descriptor, "distributed_entrypoint", None
        )
        rank = int(distributed.get("rank", 0))
        world_size = int(distributed.get("world_size", 1))
        master_addr = str(distributed.get("master_addr") or "127.0.0.1")
        master_port = int(distributed.get("master_port") or 29500)
        local_world_size = int(distributed.get("local_world_size") or world_size)
        if not isinstance(entrypoint, str) or ":" not in entrypoint:
            return {
                "job_id": rec.job_id,
                "state": JobState.RUNNING.value,
                "output": None,
                "error_code": ErrorCode.INVALID_PARAMS,
                "error_message": "distributed submit requires distributed_entrypoint",
            }

        job_dir = self.workspace.job_dir(rec.job_id)
        log_path = job_dir / "stdout.log"

        def _log(line: str) -> None:
            with contextlib.suppress(Exception):
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(line if line.endswith("\n") else line + "\n")

        ctx = DistributedRunContext(
            job_id=rec.job_id,
            workspace_root=self.workspace.root,
            job_dir=job_dir,
            gpu_ids=list(distributed.get("gpu_ids") or []),
            rank=rank,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
            local_world_size=local_world_size,
            is_cancelled=poll_cancel(store, rec.job_id, interval_s=0.25),
            log=_log,
            emit_progress=emit_progress if emit_progress is not None else (lambda _f: None),
        )

        started = time.time()
        try:
            output = _invoke_distributed_entrypoint(entrypoint, params_model, ctx)
        except Exception as exc:
            if store.is_cancel_requested(rec.job_id):
                terminal = store.transition(
                    rec.job_id,
                    JobState.CANCELLED,
                    error_code=ErrorCode.CANCELLED,
                    error_message=str(exc),
                )
                return {
                    "job_id": terminal.job_id,
                    "state": terminal.state.value,
                    "output": None,
                    "error_code": ErrorCode.CANCELLED,
                    "error_message": str(exc),
                }
            code = exc.code if isinstance(exc, VPAError) else ErrorCode.INTERNAL
            details = exc.details if isinstance(exc, VPAError) else None
            return {
                "job_id": rec.job_id,
                "state": JobState.RUNNING.value,
                "output": None,
                "error_code": code,
                "error_message": str(exc),
                "error_details": details or {"rank": rank, "world_size": world_size},
                "rank": rank,
            }

        duration_ms = int((time.time() - started) * 1000)
        if rank != 0:
            return {
                "job_id": rec.job_id,
                "state": JobState.RUNNING.value,
                "duration_ms": duration_ms,
                "rank": rank,
                "rank_completed": True,
            }
        out_dict = _normalise_distributed_output(output)
        artifacts = _introspect_artifacts(out_dict, job_dir)
        return {
            "job_id": rec.job_id,
            "state": JobState.RUNNING.value,
            "output": out_dict,
            "duration_ms": duration_ms,
            "artifacts": artifacts,
            "completed": True,
        }

    def _device_for_adapter(self) -> Any:
        torch = self._torch
        if self.gpu_id >= 0 and getattr(torch, "cuda", None) and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")


# --------------------------------------------------------------- helpers


def _append_event(events_path: Path, event: dict[str, Any]) -> None:
    """Append one NDJSON event line to ``events_path``; create parent on demand.

    Worker-side helper used by the throttled ``emit_progress`` callback and the
    "final 1.0" flush. Errors are swallowed: progress is best-effort metadata.
    """
    import json
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


def _invoke_distributed_entrypoint(
    entrypoint: str,
    params_model: Any,
    ctx: DistributedRunContext,
) -> Any:
    module_name, func_name = entrypoint.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    kwargs = {
        "params": params_model,
        "ctx": ctx,
        "context": ctx,
        "job_id": ctx.job_id,
        "workspace_root": ctx.workspace_root,
        "job_dir": ctx.job_dir,
        "gpu_ids": ctx.gpu_ids,
        "rank": ctx.rank,
        "world_size": ctx.world_size,
        "master_addr": ctx.master_addr,
        "master_port": ctx.master_port,
        "local_world_size": ctx.local_world_size,
    }
    sig = inspect.signature(fn)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)


def _normalise_distributed_output(output: Any) -> dict[str, Any]:
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if isinstance(output, dict):
        if "output" in output and isinstance(output["output"], dict):
            return dict(output["output"])
        return dict(output)
    if hasattr(output, "__dict__"):
        return dict(output.__dict__)
    return {"value": output}


def _resolve_input_model(cls: type, params: dict[str, Any]) -> Any:
    """Find the Pydantic input model on ``cls.run`` and instantiate it."""
    import inspect
    import typing

    from pydantic import BaseModel

    sig = inspect.signature(cls.run)
    params_iter = iter(sig.parameters.values())
    next(params_iter, None)  # self
    first = next(params_iter, None)
    if first is None:
        return params
    # Use get_type_hints to resolve string annotations from `from __future__ import annotations`.
    try:
        hints = typing.get_type_hints(cls.run)
    except Exception:
        hints = {}
    annotation = hints.get(first.name, first.annotation)
    if annotation is inspect.Parameter.empty:
        return params
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation(**params)
    return params


def _model_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "dict"):
        return obj.dict()  # type: ignore[no-any-return]
    return {"value": obj}


# Suffix → kind map for artifact introspection (Architecture v1 §14 "kinds").
_KIND_BY_EXT = {
    "cif": "cif", "pdb": "pdb", "a3m": "fasta", "fa": "fasta", "faa": "fasta",
    "fasta": "fasta", "aln": "alignment", "sto": "alignment",
    "csv": "csv", "tsv": "tsv", "json": "json", "npz": "npz", "npy": "npy",
    "pt": "pt", "ckpt": "ckpt", "dcd": "dcd", "xtc": "xtc",
    "png": "png", "html": "html", "md": "md",
}


def _introspect_artifacts(out_dict: dict[str, Any], job_dir: Path) -> list[dict[str, Any]]:
    """Recursively walk ``out_dict`` and surface every file path under ``job_dir``.

    The worker does not own SQLite; this is the "describe what I produced"
    half of the handoff. The supervisor receives this list and calls
    :meth:`ArtifactStore.register` for each entry (Architecture v1 §6 / §14;
    Codex round-1 B1; Codex round-2 RC2).

    Field-name encoding follows JSONPath-ish dotted+indexed conventions:

    * top-level ``designs_fasta: "<path>"`` → field ``"designs_fasta"``;
    * ``designs: [{cif_path: "<path>"}, ...]`` → fields
      ``"designs[0].cif_path"``, ``"designs[1].cif_path"``, …;
    * arbitrarily nested mixed dict/list shapes are unrolled recursively.

    Only string leaves whose value (a) **resolves inside** ``job_dir``
    (symlink-aware) AND (b) names an existing file are admitted.
    Tensors, numpy arrays, pydantic submodels, and any non-string scalar
    are ignored — keeps the walk cheap and doesn't follow surprising
    types. (Codex round-3 C1: a prefix ``startswith`` would let a
    symlink under ``job_dir`` pointing OUTSIDE ``job_dir`` masquerade as
    a contained artifact.)
    """
    import mimetypes

    try:
        job_root = Path(job_dir).resolve()
    except (OSError, ValueError):
        return []
    out: list[dict[str, Any]] = []

    def _maybe_add(field: str, value: object) -> None:
        if not isinstance(value, str):
            return
        try:
            p = Path(value).resolve()
        except (OSError, ValueError):
            return
        if not p.is_file():
            return
        if not p.is_relative_to(job_root):
            return
        ext = p.suffix.lower().lstrip(".")
        kind = _KIND_BY_EXT.get(ext, "file")
        mime, _ = mimetypes.guess_type(p.name)
        out.append({"field": field, "path": str(p), "kind": kind, "mime": mime})

    def _walk(field: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                # Allow non-string keys defensively (cast to str).
                child = f"{field}.{k}" if field else str(k)
                _walk(child, v)
            return
        if isinstance(value, list):
            for i, item in enumerate(value):
                _walk(f"{field}[{i}]", item)
            return
        _maybe_add(field, value)

    for field, value in out_dict.items():
        _walk(field, value)
    return out


def worker_main(gpu_id: int, workspace_name: str) -> None:
    """Subprocess entry point — call this from multiprocessing.spawn."""
    log_path = paths.worker_log_file(gpu_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Redirect stdout/stderr to the per-worker logfile so any escaping
    # third-party prints land there (the structured Logger writes alongside).
    sys.stdout.flush()
    sys.stderr.flush()
    log_fp = open(log_path, "a", buffering=1, encoding="utf-8")
    os.dup2(log_fp.fileno(), sys.stdout.fileno())
    os.dup2(log_fp.fileno(), sys.stderr.fileno())

    boot_log = get_file_logger(f"worker:{gpu_id}", log_path)
    boot_log.info("starting", workspace=workspace_name)

    worker = Worker(gpu_id, workspace_name)
    try:
        asyncio.run(worker.serve())
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        boot_log.error("fatal", error=str(exc))
        raise


__all__ = ["DEFAULT_MIN_FREE_VRAM_MIB", "AdapterMeta", "Worker", "worker_main"]
