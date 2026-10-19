"""
Delta manager variant that evicts every delta chunk from the OS page cache right after reading it,
so the next read of the same chunk hits storage again ("cold" switching). It uses `os.posix_fadvise(DONTNEED)`.
It is a subclass, so the core delta manager is unchanged. Linux only.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from networks.level_delta_utils import FileDeltaManager


class NoCacheFileDeltaManager(FileDeltaManager):
    def _read_chunk(self, start, end):
        if not hasattr(os, "posix_fadvise"):
            raise RuntimeError("posix_fadvise is not available on this platform (Linux only)")
        pos = self._data_start + start
        if self._file.tell() != pos:
            self._file.seek(pos)
        size = end - start
        if size < 0:
            size = -1
        data = self._file.read(size)
        try:
            os.posix_fadvise(self._file.fileno(), pos, len(data), os.POSIX_FADV_DONTNEED)
        except OSError as e:  # e.g. filesystem without fadvise support
            print(f"[warn] posix_fadvise failed: {e}")
        return data


@contextmanager
def file_delta_manager_nocache(path, managed_config):
    with open(path, "rb") as f:
        yield NoCacheFileDeltaManager(f, managed_config)
