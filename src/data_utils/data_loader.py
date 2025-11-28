import pandas as pd
from pathlib import Path


SAMPLING_FREQUENCY: float = 100.  # Hz


def load_data(
    motion_file_path: str | Path,
    tracked_points: None | str | list[str] = None
) -> pd.DataFrame:
    """
    Load motion capture data from a TSV file.

    Parameters
    ----------
    motion_file_path : str | Path
        Path to the TSV file containing motion capture data.
    tracked_points : None | str | list[str], optional
        Specific points to load from the data. If None, all points are loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded motion capture data.
    """
    # Parse the file path
    motion_file_path = Path(motion_file_path) if isinstance(motion_file_path, str) else motion_file_path

    # Load the associated metadata
    channels_df = pd.read_csv(
        motion_file_path.parent / motion_file_path.name.replace("_motion.tsv", "_channels.tsv"),
        header=0, 
        sep="\t"
    )
    fs = channels_df["sampling_frequency"].values[0].astype(float)
    units = channels_df[channels_df["type"]=="POS"]["units"].values[0]

    # Load the TSV file into a DataFrame
    motion_df = pd.read_csv(motion_file_path, sep="\t", header=0)
    motion_df = motion_df[[c for c in motion_df.columns if c.split("_")[-1] not in ["n/a", "err"]]]
    if fs != SAMPLING_FREQUENCY:
        motion_df = motion_df.iloc[::int(fs / SAMPLING_FREQUENCY), :].reset_index(drop=True)
    if tracked_points is not None:
        tracked_points = [tracked_points] if isinstance(tracked_points, str) else tracked_points
        motion_df = motion_df[[c for c in motion_df.columns if "_".join(c.split("_")[:-2]) in tracked_points]]
    if units == "mm":
        motion_df /= 1000.0  # convert mm to m
    elif units == "cm":
        motion_df /= 100.0  # convert cm to m
    else:
        pass  # assume units are already in meters
    return motion_df