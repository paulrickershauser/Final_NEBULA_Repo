import pickle, numpy as np
from collections import defaultdict

obs = "SBSS (USA 216)"

with open(r"C:\\Users\\prick\\Desktop\\Research\\NEBULA\\NEBULA_OUTPUT\\STARS\\GAIA_DR3_G\\obs_gaia_cones.pkl","rb") as f:
    gaia = pickle.load(f)

with open(r"C:\Users\prick\Desktop\Research\NEBULA\NEBULA_OUTPUT\STARS\GAIA_DR3_G\obs_star_projections.pkl","rb") as f:
    proj = pickle.load(f)

gaia_ids64 = gaia[obs]["windows"][0]["gaia_source_id"]          # should be int64 ~1e18
gaia_ids32 = gaia_ids64.astype(np.int32)                         # truncates/overflows

print("GAIA rows:", len(gaia_ids64))
print("Unique after int32:", len(np.unique(gaia_ids32)))

# show a few collisions (same int32 -> multiple real Gaia IDs)
groups = defaultdict(list)
for full, low in zip(gaia_ids64, gaia_ids32):
    groups[int(low)].append(int(full))

collisions = [(k,v) for k,v in groups.items() if len(v) > 1]
print("Collision buckets:", len(collisions))
print("Example collision:", collisions[0])

# compare to your projection dict keys
proj_keys = list(proj[obs]["windows"][0]["stars"].keys())
print("Projection dict len:", len(proj_keys))
print("Projection key range (int):", min(map(int, proj_keys)), max(map(int, proj_keys)))
