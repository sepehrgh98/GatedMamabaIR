import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    "/speed-scratch/se_gham/GatedMambaIR/urban100_selectivescan_metrics.npy",
    allow_pickle=True
)

# data is an array of dicts
corr_change = np.array(
    [d["corr_change_frob"] for d in data],
    dtype=np.float32
)

print("Mean correlation change:", corr_change.mean())
print("Std  correlation change:", corr_change.std())

# -----------------------
# SAVE plot instead of show
# -----------------------
plt.figure(figsize=(6,4))
plt.hist(corr_change, bins=30)
plt.title("Urban100: ||R_in - R_out|| after SelectiveScan")
plt.xlabel("Frobenius norm")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("urban100_selectivescan_corr_change.png", dpi=200)
plt.close()

print("Saved: urban100_selectivescan_corr_change.png")
