import os
import json
import argparse

from dataclasses import dataclass

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as TF


from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_text_inference.to(device)


@torch.no_grad()
def compute_clip_scores(df):
    scores = []
    for idx, (row) in tqdm(df.iterrows(), total=len(df)):
        scores.append(
            image_text_inference.get_similarity_score_from_raw_data(
                Path(row["path"]), row["report"]
            )
        )
    return scores


@dataclass(frozen=True)
class PairSpec:
    name: str
    label: str
    a_df: pd.DataFrame
    control: bool = False


def sample_two_disjoint(a, b, rng):
    b = b.sample(n=len(a), random_state=rng).reset_index(drop=True)
    neg_report = b["report"].tolist()
    a["report"] = neg_report
    return a


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
    parser = argparse.ArgumentParser(description="Compute BioVIL CLIP-score pairs.")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="biomedvil_results.json",
        help="JSON output path (relative to repo root).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by loading existing JSON and continuing repeats.",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP-score computation.",
    )
    parser.add_argument(
        "--biovil-ckpt",
        type=str,
        default=None,
        help="Path to BioViL image encoder checkpoint.",
    )
    parser.add_argument(
        "--biovil-model-id",
        type=str,
        default="microsoft/BiomedVLP-BioViL-T",
        help="HuggingFace model id for BioViL text model.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    measurement_df_real = pd.read_csv("anatomical_plausibility_signals.csv")
    measurement_df_cheff = pd.read_csv("anatomical_plausibility_signals_cheff.csv")
    measurement_df_roentgen = pd.read_csv(
        "anatomical_plausibility_signals_roentgen.csv"
    )

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

    pairs = [
        PairSpec(
            name="real",
            label="no_finding",
            a_df=nofinding_cases_real,
        ),
        PairSpec(
            name="real",
            label="cardiomegaly",
            a_df=cardiomegaly_cases_real,
        ),
        PairSpec(
            name="cheff",
            label="no_finding",
            a_df=nofinding_cases_cheff,
        ),
        PairSpec(
            name="cheff",
            label="cardiomegaly",
            a_df=cardiomegaly_cases_cheff,
        ),
        PairSpec(
            name="roentgen",
            label="no_finding",
            a_df=nofinding_cases_roentgen,
        ),
        PairSpec(
            name="roentgen",
            label="cardiomegaly",
            a_df=cardiomegaly_cases_roentgen,
        ),
        PairSpec(
            name="real_vs_real",
            label="no_finding_cardiomegaly",
            a_df=sample_two_disjoint(
                cardiomegaly_cases_real, nofinding_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding",
            a_df=sample_two_disjoint(
                nofinding_cases_cheff, nofinding_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_cheff",
            label="cardiomegaly",
            a_df=sample_two_disjoint(
                cardiomegaly_cases_cheff, cardiomegaly_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding",
            a_df=sample_two_disjoint(
                nofinding_cases_roentgen, nofinding_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="cardiomegaly",
            a_df=sample_two_disjoint(
                cardiomegaly_cases_roentgen, cardiomegaly_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_cheff",
            label="no_finding_cardiomegaly",
            a_df=sample_two_disjoint(
                cardiomegaly_cases_cheff, nofinding_cases_real, args.seed
            ),
        ),
        PairSpec(
            name="real_vs_roentgen",
            label="no_finding_cardiomegaly",
            a_df=sample_two_disjoint(
                cardiomegaly_cases_roentgen, nofinding_cases_real, args.seed
            ),
        ),
    ]

    results = {}
    if args.resume:
        results = load_existing_json(args.output)

    for pair in pairs:
        desc = f"{pair.name}:{pair.label}"
        key = f"{pair.name}__{pair.label}"
        print(f"Processing pair {desc}...")
        if key not in results:
            results[key] = {
                "pair_name": pair.name,
                "label": pair.label,
                "n_samples": args.n_samples,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "device": device,
                "clip_scores": [],
            }

        clip_a = compute_clip_scores(
            pair.a_df,
        )

        results[key]["clip_scores"].extend(clip_a)

        # persist after each pair to avoid losing progress
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
