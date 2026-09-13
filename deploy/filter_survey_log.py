import sys
path = "/tmp/amcl_survey_20260827.log"
needles = (
    "ERROR", "FATAL", "failed", "survey", "stale", "timeout",
    "RESULT_", "safety_stop", "offline_recorder", "healthy",
    "nvargus", "OPERATION", "Couldn't", "required",
)
with open(path, "r") as stream:
    for line in stream:
        if any(item in line for item in needles):
            sys.stdout.write(line[:400] + ("\n" if not line.endswith("\n") else ""))
