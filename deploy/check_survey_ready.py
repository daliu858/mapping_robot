import glob
logs = sorted(glob.glob("/home/jetbot/.ros/log/*/offline_recorder*.log"))
path = logs[-1]
print("LOG", path)
with open(path) as stream:
    lines = stream.readlines()
for line in lines:
    if "survey" in line or "FATAL" in line or "healthy" in line or "recording to" in line:
        print(line.rstrip()[:240])
print("BAGS")
import os
for name in sorted(os.listdir("/home/jetbot")):
    if name.startswith("semantic_survey_home_20260827"):
        print(name)
