import glob
import os, csv
import pandas as pd
from tbparse import SummaryReader

# Finds all event.out.tfevents files in runs_dir, returns a list of sull fule paths to event files
def find_event_files(runs_dir):
    pattern = os.path.join(runs_dir, "**", "events.out.tfevents*")
    return glob.glob(pattern, recursive=True)

# Reads all the event files in runs_dir, filters by tag and writes their scalars to CSV files in out_dir
def convert_all(runs_dir, out_dir, tags=None):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Discover event files
    event_paths = find_event_files(runs_dir)
    if not event_paths:
        print(f"⚠️  No event files found in {runs_dir}")
        return

    # Build a SummaryReader for each event file
    readers = [SummaryReader(path) for path in event_paths]

    # Concatenate every scalar DataFrame into one big DataFrame, each r.scalars has columns: ['wall_time','step','tag','value']
    df = pd.concat((r.scalars for r in readers), ignore_index=True)

    # If a tag filter is provided, apply it
    if tags is not None:
        df = df[df.tag.isin(tags)]

    # For each distinct tag, write out a CSV of (step,value)
    for tag, group in df.groupby("tag"):
        # sanitize tag for filename, e.g. "Loss/train" -> "loss_train.csv"
        safe_tag = tag.lower().replace("/", "_")

        out = group.sort_values("step")[["step", "value"]].copy()
        out["step"] = out["step"] + 1

        csv_name = f"{safe_tag}.csv"
        csv_path = os.path.join(out_dir, csv_name)

        # no index column, just two columns: step,value
        out.to_csv(csv_path, index=False)
        print(f"→ Wrote {csv_path}")

if __name__ == "__main__":
    # Name of the project subfolder under `runs/`
    project_name = "test_2"
    runs_dir = os.path.join("runs", project_name)

    # Where you'd like the CSVs to land in the backend folder
    out_dir = os.path.join("..", "backend", "data", "metrics_csv", project_name)

    # Which tags to export. Use None to grab all tags.
    tags = ["Loss/train", "Accuracy/train", "FLOPs/model", "Parameters/model", "Energy/train_kwh", "Time/train"] 

    # Kick off the conversion
    convert_all(runs_dir, out_dir, tags)