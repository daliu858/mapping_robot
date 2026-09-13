# -*- coding: utf-8 -*-
"""PC-side loader for the canonical implementation shipped on the Jetson."""
from __future__ import print_function

import importlib.util
import os


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE = os.path.join(
    _ROOT, "src", "jetbot_pro", "scripts", "map_identity.py")
_SPEC = importlib.util.spec_from_file_location("_slam_car_map_identity", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MANIFEST_SCHEMA = _MODULE.MANIFEST_SCHEMA
YAW_EPSILON = _MODULE.YAW_EPSILON
is_sha256 = _MODULE.is_sha256
load_map_identity = _MODULE.load_map_identity
quaternion_yaw = _MODULE.quaternion_yaw
require_zero_grid_yaw = _MODULE.require_zero_grid_yaw
require_zero_origin_yaw = _MODULE.require_zero_origin_yaw
verify_identity_fields = _MODULE.verify_identity_fields
