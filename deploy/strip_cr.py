#!/usr/bin/env python
import glob
import os
import sys

paths = sys.argv[1:]
if not paths:
    paths = []
    for pattern in ("/tmp/*.sh", "/tmp/*.py", "/tmp/*.launch", "/tmp/*.yaml"):
        paths.extend(glob.glob(pattern))
for path in paths:
    if not os.path.isfile(path):
        continue
    data = open(path, "rb").read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    open(path, "wb").write(data)
    if path.endswith(".sh"):
        os.chmod(path, 0o755)
print("STRIP_OK", len(paths))
