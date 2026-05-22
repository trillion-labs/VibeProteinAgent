from __future__ import annotations


class WorkflowRunner:
    """Marker class for the job-polling responsibility held by WorkflowExecutor.

    The public ``WorkflowExecutor`` facade still owns dispatch to preserve
    imports and behavior. This class gives the package a named home for future
    runner extraction without changing the stable API in this polish pass.
    """


__all__ = ["WorkflowRunner"]
