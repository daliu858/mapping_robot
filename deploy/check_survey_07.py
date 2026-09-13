#!/usr/bin/env python3
import glob
import os

logs = sorted(glob.glob("/home/jetbot/.ros/log/*/offline_recorder-*.log"))
print("LOG", logs[-1] if logs else None)
if logs:
    print(open(logs[-1]).read()[-2500:])
print("---")
print("ACTIVE", os.path.exists("/home/jetbot/semantic_survey_home_20260827_07.bag.active"))
print("COMPLETE", os.path.exists("/home/jetbot/semantic_survey_home_20260827_07.bag.complete.json"))
print("FAILED", os.path.exists("/home/jetbot/semantic_survey_home_20260827_07.bag.failed.json"))
print("BAG", os.path.exists("/home/jetbot/semantic_survey_home_20260827_07.bag"))
