import glob
import os

bags = sorted(glob.glob("/home/jetbot/semantic_survey_home_20260827_02.bag*"))
print("BAGS", bags or "none")
logs = sorted(glob.glob("/home/jetbot/.ros/log/*/offline_recorder*.log"))
print("LOGS", logs[-3:] if logs else "none")
if logs:
    with open(logs[-1]) as stream:
        text = stream.read()
    print("---- recorder log tail ----")
    print(text[-3000:])
launch = "/tmp/amcl_survey_20260827.log"
if os.path.isfile(launch):
    with open(launch) as stream:
        text = stream.read()
    print("---- launch size", len(text), "----")
    for line in text.splitlines():
        if "survey" in line.lower() or "FATAL" in line or "REQUIRED" in line:
            print(line[:300])
