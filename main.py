import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from src.data_utils import data_loader


DATA_PATH = "./data/sub-pp002/motion/sub-pp002_task-treadmill_tracksys-omc_motion.tsv"
fs = data_loader.SAMPLING_FREQUENCY


def main():
    tracked_points = [
        f"{s}_{mrk}"
        for mrk in ["heel", "toe", "asis", "psis"]
        for s in ["l", "r"]
    ]
    motion_df = data_loader.load_data(
        DATA_PATH,
        tracked_points=tracked_points
    )

    # Compute a virtual mid PSIS and mid ASIS marker
    for x in ["x", "y", "z"]:
        motion_df[f"mid_psis_POS_{x}"] = 0.5 * (
            motion_df["l_psis_POS_" + x] + motion_df["r_psis_POS_" + x]
        )
        motion_df[f"mid_asis_POS_{x}"] = 0.5 * (
            motion_df["l_asis_POS_" + x] + motion_df["r_asis_POS_" + x]
        )

    # Identify heel strikes as the minima of the heel marker position relative to mid PSIS
    x = motion_df["l_heel_POS_x"] - motion_df["mid_psis_POS_x"]
    LHS, _ = scipy.signal.find_peaks(-x, distance=fs*0.5)

    # Identify toe offs as the maxima of the toe marker position relative to mid PSIS 
    y = motion_df["l_toe_POS_x"] - motion_df["mid_psis_POS_x"]
    LTO, _ = scipy.signal.find_peaks(y, distance=fs*0.5)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(x)
    ax.plot(LHS, x[LHS], "x", label="LHS")
    ax.plot(y)
    ax.plot(LTO, y[LTO], "o", label="LTO")
    ax.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
