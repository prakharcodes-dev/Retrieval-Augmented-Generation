"""
Operation state management for RAG Application lifecycle.
Enables transparent tracking of subsystem states and graceful recovery after failure.
"""

from enum import Enum
from typing import Optional, Dict, Any
import time


class OperationState(str, Enum):
    READY = "READY"
    VALIDATING = "VALIDATING"
    INGESTING = "INGESTING"
    RETRIEVING = "RETRIEVING"
    GENERATING = "GENERATING"
    RECOVERING = "RECOVERING"
    DEGRADED = "DEGRADED"


class StateTracker:
    """Tracks application state transitions and last error details."""

    def __init__(self, initial_state: OperationState = OperationState.READY):
        self._state = initial_state
        self._last_error: Optional[str] = None
        self._last_updated: float = time.time()
        self._metadata: Dict[str, Any] = {}

    @property
    def current_state(self) -> OperationState:
        return self._state

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def transition_to(self, new_state: OperationState, error: Optional[str] = None, **metadata: Any) -> None:
        self._state = new_state
        if error is not None:
            self._last_error = error
        self._last_updated = time.time()
        if metadata:
            self._metadata.update(metadata)

    def is_healthy(self) -> bool:
        return self._state in (OperationState.READY, OperationState.VALIDATING, OperationState.INGESTING, OperationState.RETRIEVING, OperationState.GENERATING)

    def reset(self) -> None:
        self._state = OperationState.READY
        self._last_error = None
        self._last_updated = time.time()
        self._metadata.clear()
