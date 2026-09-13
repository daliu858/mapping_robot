import glob, os, json
print("FAILED JSON")
for path in sorted(glob.glob("/home/jetbot/semantic_survey_home_20260827_0*.bag.failed.json")):
    with open(path) as stream:
        data = json.load(stream)
    print(os.path.basename(path), data.get("reason"))
print("RECORDER LOGS")
for path in sorted(glob.glob("/home/jetbot/.ros/log/*/offline_recorder*.log"), key=os.path.getmtime):
    print(int(os.path.getmtime(path)), path, os.path.getsize(path))
logs = sorted(glob.glob("/home/jetbot/.ros/log/*/offline_recorder*.log"), key=os.path.getmtime)
if logs:
    path = logs[-1]
    print("LATEST", path)
    with open(path) as stream:
        lines = [line for line in stream if "survey" in line or "nomotion" in line]
    for line in lines[-15:]:
        print(line.rstrip()[:250])
