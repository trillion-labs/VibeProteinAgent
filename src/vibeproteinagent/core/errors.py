from __future__ import annotations

from typing import Any


class ErrorCode:
    UNKNOWN_TOOL = "E_UNKNOWN_TOOL"
    INVALID_PARAMS = "E_INVALID_PARAMS"
    INVALID_OUTPUT = "E_INVALID_OUTPUT"
    NO_GPU = "E_NO_GPU"
    VRAM_EXHAUSTED = "E_VRAM_EXHAUSTED"
    TIMEOUT = "E_TIMEOUT"
    CANCELLED = "E_CANCELLED"
    DEPENDENCY = "E_DEPENDENCY"
    INTERNAL = "E_INTERNAL"
    DAEMON_UNAVAILABLE = "E_DAEMON_UNAVAILABLE"
    DAEMON_WORKSPACE_MISMATCH = "E_DAEMON_WORKSPACE_MISMATCH"
    WORKSPACE = "E_WORKSPACE"
    NOT_FOUND = "E_NOT_FOUND"
    CONFLICT = "E_CONFLICT"
    BUSY = "E_BUSY"


# Error codes whose jobs are eligible for auto-retry (Architecture v1 §5).
# Domain errors (E_INVALID_PARAMS, E_UNKNOWN_TOOL, ...) are deliberately
# excluded — they will never be magically retryable.
RETRYABLE_ERRORS: frozenset[str] = frozenset({
    ErrorCode.TIMEOUT,
    ErrorCode.NO_GPU,
    ErrorCode.VRAM_EXHAUSTED,
    ErrorCode.DAEMON_UNAVAILABLE,
})


EXIT_USER_ERROR = 1
EXIT_RUNTIME_ERROR = 2
EXIT_DAEMON_ERROR = 3

_EXIT_MAP: dict[str, int] = {
    ErrorCode.UNKNOWN_TOOL: EXIT_USER_ERROR,
    ErrorCode.INVALID_PARAMS: EXIT_USER_ERROR,
    ErrorCode.INVALID_OUTPUT: EXIT_RUNTIME_ERROR,
    ErrorCode.NOT_FOUND: EXIT_USER_ERROR,
    ErrorCode.CONFLICT: EXIT_USER_ERROR,
    ErrorCode.WORKSPACE: EXIT_USER_ERROR,
    ErrorCode.NO_GPU: EXIT_RUNTIME_ERROR,
    ErrorCode.VRAM_EXHAUSTED: EXIT_RUNTIME_ERROR,
    ErrorCode.TIMEOUT: EXIT_RUNTIME_ERROR,
    ErrorCode.CANCELLED: EXIT_RUNTIME_ERROR,
    ErrorCode.DEPENDENCY: EXIT_RUNTIME_ERROR,
    ErrorCode.INTERNAL: EXIT_RUNTIME_ERROR,
    ErrorCode.BUSY: EXIT_RUNTIME_ERROR,
    ErrorCode.DAEMON_UNAVAILABLE: EXIT_DAEMON_ERROR,
    ErrorCode.DAEMON_WORKSPACE_MISMATCH: EXIT_USER_ERROR,
}


def exit_code_for(error_code: str) -> int:
    return _EXIT_MAP.get(error_code, EXIT_RUNTIME_ERROR)


class VPAError(Exception):
    code: str = ErrorCode.INTERNAL

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class UnknownToolError(VPAError):
    code = ErrorCode.UNKNOWN_TOOL


class InvalidParamsError(VPAError):
    code = ErrorCode.INVALID_PARAMS


class NotFoundError(VPAError):
    code = ErrorCode.NOT_FOUND


class ConflictError(VPAError):
    code = ErrorCode.CONFLICT


class WorkspaceError(VPAError):
    code = ErrorCode.WORKSPACE


class DaemonUnavailableError(VPAError):
    code = ErrorCode.DAEMON_UNAVAILABLE


class DaemonWorkspaceMismatchError(VPAError):
    code = ErrorCode.DAEMON_WORKSPACE_MISMATCH


class DependencyError(VPAError):
    code = ErrorCode.DEPENDENCY


class TimeoutError_(VPAError):
    code = ErrorCode.TIMEOUT


class CancelledError(VPAError):
    code = ErrorCode.CANCELLED


class NoGpuError(VPAError):
    code = ErrorCode.NO_GPU


class VramExhaustedError(VPAError):
    code = ErrorCode.VRAM_EXHAUSTED


class BusyError(VPAError):
    code = ErrorCode.BUSY
