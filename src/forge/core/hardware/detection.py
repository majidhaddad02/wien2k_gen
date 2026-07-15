"""SysFSHardwareInfo: concrete hardware detection implementation."""

from .cpu import CpuDetectionMixin
from .system import SystemDetectionMixin
from .types import HardwareInfoProvider


class SysFSHardwareInfo(CpuDetectionMixin, SystemDetectionMixin, HardwareInfoProvider):
    """Concrete hardware info provider via Linux sysfs + /proc/cpuinfo."""


__all__ = ["SysFSHardwareInfo"]
