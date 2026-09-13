# -*- coding: utf-8 -*-
"""Read Melodic bags that Python ``rosbag.Bag(..., "a")`` later appended to.

``rosbags`` caches INDEX_DATA header field offsets from the first index
record. C++ ``rosbag record`` writes ``conn`` first; rospy's append path
can write ``count`` first. The cached offsets then make
``size == count * 12`` fail, so the bag looks corrupt even though the
chunks are intact.

Clearing the cached offsets before every index record restores a
per-record parse. Call ``patch_rosbags_index_headers()`` before
``AnyReader``.
"""
from __future__ import print_function

_ORIGINAL = None
_PATCHED = False


def patch_rosbags_index_headers():
    """Make ``rosbags`` reparse each INDEX_DATA header independently."""
    global _ORIGINAL, _PATCHED
    from rosbags.rosbag1.reader import Reader

    if _ORIGINAL is None:
        _ORIGINAL = Reader.read_index_data
    if _PATCHED:
        return
    original = _ORIGINAL

    def read_index_data(self, pos, indexes):
        self.index_data_header_offsets = None
        return original(self, pos, indexes)

    Reader.read_index_data = read_index_data
    _PATCHED = True
    return
