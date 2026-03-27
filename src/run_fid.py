import argparse
import os
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as TF
from tqdm import tqdm


class DatasetFID(Dataset):
    def __init__(self, df, size=(256, 256)):
        self.df = df
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["path"]
        image = read_image(file_name, mode=ImageReadMode.RGB)
        image = TF.resize(image, self.size)
        return image


def compute_fid(
    real_images_df, synth_images_df, batch_size=32, num_workers=0, device="cuda"
):
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

    real_dataset = DatasetFID(df=real_images_df)
    synth_dataset = DatasetFID(df=synth_images_df)

    real_dataloader = DataLoader(
        real_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    synth_dataloader = DataLoader(
        synth_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    with torch.no_grad():
        for real_batch in real_dataloader:
            fid.update(real_batch.to(device), real=True)

        for synth_batch in synth_dataloader:
            fid.update(synth_batch.to(device), real=False)

    fid_score = fid.compute().item()
    return fid_score


@dataclass(frozen=True)
class PairSpec:
    name: str
    label: str
    a_name: str
    b_name: str
    a_df: pd.DataFrame
    b_df: pd.DataFrame
    control: bool = False


def sample_two_disjoint(df, n_samples, rng):
    if len(df) < 2 * n_samples:
        raise ValueError(f"Need >= {2 * n_samples} rows, got {len(df)}")
    idx = rng.choice(len(df), size=2 * n_samples, replace=False)
    a_idx = idx[:n_samples]
    b_idx = idx[n_samples:]
    a = df.iloc[a_idx].reset_index(drop=True)
    b = df.iloc[b_idx].reset_index(drop=True)
    return a, b


def sample_one(df, n_samples, rng):
    if len(df) < n_samples:
        raise ValueError(f"Need >= {n_samples} rows, got {len(df)}")
    idx = rng.choice(len(df), size=n_samples, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def load_existing_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compute FID pairs overnight.")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="fid_results.json",
        help="JSON output path (relative to repo root).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by loading existing JSON and continuing repeats.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    measurement_df_real = pd.read_csv("anatomical_plausibility_signals.csv")
    measurement_df_cheff = pd.read_csv("anatomical_plausibility_signals_cheff.csv")
    measurement_df_roentgen = pd.read_csv(
        "anatomical_plausibility_signals_roentgen.csv"
    )
    measurement_df_chexpert = pd.read_csv("morphometric_measurements_chexpert.csv")

    nofinding_cases_real = measurement_df_real[
        measurement_df_real["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_real = measurement_df_real[
        measurement_df_real["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]

    nofinding_cases_cheff = measurement_df_cheff[
        measurement_df_cheff["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_cheff = measurement_df_cheff[
        measurement_df_cheff["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]

    nofinding_cases_roentgen = measurement_df_roentgen[
        measurement_df_roentgen["prompt"].str.contains("healthy", na=False, case=False)
    ]
    cardiomegaly_cases_roentgen = measurement_df_roentgen[
        measurement_df_roentgen["prompt"].str.contains(
            "with cardiomegaly", na=False, case=False
        )
    ]
    no_finding_cases_chexpert = measurement_df_chexpert[
        measurement_df_chexpert["No Finding"] == 1.0
    ]

    cardiomegaly_cases_chexpert = measurement_df_chexpert[
        measurement_df_chexpert["Cardiomegaly"] == 1.0
    ]

    pairs = [
        PairSpec(
            name="real_vs_chexpert",
            label="no_finding",
            a_name="real",
            b_name="chexpert",
            a_df=nofinding_cases_real,
            b_df=no_finding_cases_chexpert,
            control=False,
        ),
        PairSpec(
            name="real_vs_chexpert",
            label="cardiomegaly",
            a_name="real",
            b_name="chexpert",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_chexpert,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding",
            a_name="real",
            b_name="cheff",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding",
            a_name="real",
            b_name="roentgen",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_roentgen,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="cardiomegaly",
            a_name="real",
            b_name="cheff",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="cardiomegaly",
            a_name="real",
            b_name="roentgen",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_roentgen,
            control=False,
        ),
        PairSpec(
            name="real_vs_real",
            label="no_finding",
            a_name="real",
            b_name="real",
            a_df=nofinding_cases_real,
            b_df=nofinding_cases_real,
            control=True,
        ),
        PairSpec(
            name="real_vs_real",
            label="cardiomegaly",
            a_name="real",
            b_name="real",
            a_df=cardiomegaly_cases_real,
            b_df=cardiomegaly_cases_real,
            control=True,
        ),
        PairSpec(
            name="real_vs_real",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="real",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_real,
            control=False,
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="cheff",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_cheff,
            control=False,
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding_cardiomegaly",
            a_name="real",
            b_name="roentgen",
            a_df=nofinding_cases_real,
            b_df=cardiomegaly_cases_roentgen,
            control=False,
        ),
    ]

    rng = np.random.default_rng(args.seed)

    results = {}
    if args.resume:
        results = load_existing_json(args.output)

    for pair in pairs:
        desc = f"{pair.name}:{pair.label}"
        key = f"{pair.name}__{pair.label}"
        if key not in results:
            results[key] = {
                "pair_name": pair.name,
                "label": pair.label,
                "a_name": pair.a_name,
                "b_name": pair.b_name,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "fid_scores": [],
            }
        existing_scores = results[key]["fid_scores"]
        start_idx = len(existing_scores)
        for repeat_idx in tqdm(range(args.n_repeats), desc=desc):
            if repeat_idx < start_idx:
                continue

            if pair.control:
                real_df, synth_df = sample_two_disjoint(pair.a_df, args.n_samples, rng)
            else:
                real_df = sample_one(pair.a_df, args.n_samples, rng)
                synth_df = sample_one(pair.b_df, args.n_samples, rng)

            fid_score = compute_fid(
                real_df,
                synth_df,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            results[key]["fid_scores"].append(fid_score)

        # persist after each pair to avoid losing progress
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
