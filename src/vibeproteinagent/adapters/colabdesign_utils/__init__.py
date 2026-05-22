from .adapter import (
    ColabDesignUtilsAdapter,
    ColabDesignUtilsInput,
    ColabDesignUtilsOutput,
    colabdesign_runtime_skip_reason,
)


def availability_skip_reason() -> str | None:
    return colabdesign_runtime_skip_reason()

__all__ = [
    "ColabDesignUtilsAdapter",
    "ColabDesignUtilsInput",
    "ColabDesignUtilsOutput",
    "availability_skip_reason",
    "colabdesign_runtime_skip_reason",
]
