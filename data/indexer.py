import os
import json

files = [int(i[:-5]) for i in sorted(list(os.listdir("retrained")))]
with open(os.path.join("retrained", "index.json"), "w") as f:
    json.dump({"indices": files}, f, indent=4)