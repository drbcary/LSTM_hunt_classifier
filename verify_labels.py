import pandas as pd
import sys
sys.path.insert(0, '.')
import dataset_utils

# Test against the example file
hunts = dataset_utils.load_dan_labels('example_files')
print(f"Keys found: {list(hunts.keys())}")
for key, hunt_list in hunts.items():
    print(f"\n{key}: {len(hunt_list)} hunts")
    for i, h in enumerate(hunt_list[:3]):
        print(f"  Hunt {i}: enter={h['enter']}, pursuit={h['pursuit_start']}, capture={h['capture']}, cons_end={h['consumption_end']}")
