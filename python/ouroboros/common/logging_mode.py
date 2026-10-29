"""The operation logging policy shared by tasks, pipelines, and input adapters."""

from enum import Enum


class LoggingMode(str, Enum):
    OFF = "off"
    STANDARD = "standard"
    DETAILED = "detailed"

    def __str__(self) -> str:
        return self.value

    @property
    def reporting_enabled(self) -> bool:
        """Reports and optional step statistics have the same collection policy."""
        return self is not LoggingMode.OFF

    @property
    def memory_enabled(self) -> bool:
        return self is LoggingMode.DETAILED
