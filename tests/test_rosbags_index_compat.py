"""Regression: mixed INDEX_DATA field order after rospy bag append."""
from __future__ import print_function

import importlib.util
import os
import struct
import unittest
from collections import defaultdict
from io import BytesIO


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "rosbags_index_compat.py")
SPEC = importlib.util.spec_from_file_location("rosbags_index_compat", SCRIPT)
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)


def _field(name, value):
    body = name.encode("ascii") + b"=" + value
    return struct.pack("<I", len(body)) + body


def index_record(order, conn, count):
    fields = {
        "conn": _field("conn", struct.pack("<I", conn)),
        "count": _field("count", struct.pack("<I", count)),
        "ver": _field("ver", struct.pack("<I", 1)),
        "op": _field("op", struct.pack("B", 4)),
    }
    header = b"".join(fields[name] for name in order)
    if len(header) != 47:
        raise AssertionError("INDEX_DATA header must be 47 bytes")
    payload = b"\x00" * (count * 12)
    return (struct.pack("<I", 47) + header +
            struct.pack("<I", len(payload)) + payload)


def _reader_with_bytes(payload):
    from rosbags.rosbag1.reader import Reader

    reader = Reader.__new__(Reader)
    reader.bio = BytesIO(payload)
    reader.index_data_header_offsets = None
    return reader


class RosbagsIndexCompatTest(unittest.TestCase):
    def test_cached_offsets_break_count_first_headers(self):
        from rosbags.rosbag1.reader import Reader

        original = COMPAT._ORIGINAL or Reader.read_index_data
        mixed = (
            index_record(("conn", "count", "op", "ver"), 0, 1) +
            index_record(("count", "ver", "conn", "op"), 16, 1))
        reader = _reader_with_bytes(mixed)
        indexes = defaultdict(list)
        original(reader, 0, indexes)
        with self.assertRaises(AssertionError):
            original(reader, 0, indexes)

    def test_patch_reads_conn_first_and_count_first_indexes(self):
        COMPAT.patch_rosbags_index_headers()
        mixed = (
            index_record(("conn", "count", "op", "ver"), 0, 1) +
            index_record(("count", "ver", "conn", "op"), 16, 1))
        reader = _reader_with_bytes(mixed)
        indexes = defaultdict(list)
        reader.read_index_data(0, indexes)
        reader.read_index_data(0, indexes)
        self.assertEqual(len(indexes[0]), 1)
        self.assertEqual(len(indexes[16]), 1)


if __name__ == "__main__":
    unittest.main()
