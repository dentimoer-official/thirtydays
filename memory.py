"""프로세스 메모리 측정 (macOS).

활동 모니터의 "메모리"와 같은 phys_footprint를 쓴다. 통합 메모리라 MLX(GPU) 할당도 포함된다.
peak()는 프로세스가 시작된 뒤 가장 많이 쓴 순간의 값이다.
"""

import ctypes
import ctypes.util
import os

GB = 1024 ** 3
MB = 1024 ** 2

RUSAGE_INFO_V4 = 4


class RusageInfoV4(ctypes.Structure):
    # <sys/resource.h>의 struct rusage_info_v4와 같은 순서
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[(name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )],
    ]


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _rusage():
    info = RusageInfoV4()
    if _libc.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")

    return info


def current():
    return _rusage().ri_phys_footprint


def peak():
    return _rusage().ri_lifetime_max_phys_footprint
