import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
from wand.measurements import extract_morphometrics
from wand.segmentation import read_image, transform_img, load_segmentation_model

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--csv_r",
        type=str,
        help="Path to the CSV file containing real image paths and metadata.",
    )
    argparser.add_argument(
        "--csv_s",
        type=str,
        help="Path to the CSV file containing synthetic image paths and metadata.",
    )
    
    argparser.add_argument(
        "--num", type=int, default=10000, help="Number of images to process."
    )
    argparser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save the output CSV."
    )
    argparser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Number of images to process before saving a checkpoint.",
    )   
    args = argparser.parse_args()

    CHECKPOINT_INTERVAL = args.checkpoint_interval


    # load segmentation 
    seg_model = load_segmentation_model()

    # 1. LOAD METADATA
    real_metadata = pd.read_csv(args.csv_real)
    synthetic_metadata = pd.read_csv(args.csv_s)

    def wand_pipeline(metadata: pd.DataFrame, save_as: str = 'real_features.csv'):
        # 1. CHECKPOINT:Load existing data or initialize new DataFrame
        OUTPUT_FILENAME = save_as
        OUTPUT_PATH = os.path.join(args.output_dir, OUTPUT_FILENAME)
        
        if os.path.exists(OUTPUT_PATH):
            print(f"Found existing data at {OUTPUT_PATH}. Resuming...")
            df = pd.read_csv(OUTPUT_PATH)
            start_index = len(df)
            print(
                f"   -> Already processed {start_index} images. Starting from index {start_index}."
            )
        else:
            print("No existing data found. Starting new feature extraction.")
            df = pd.DataFrame()
            processed_ids = set()
            start_index = 0

        total_images = min(args.num, len(metadata))

        for i in tqdm(range(start_index, total_images), total=total_images):

            value = metadata.iloc[i]
            image_path = value[
                "Path"
            ]  # Assuming the CSV has a column named 'Path' with the image file paths, rename if necessary
            try:
                # Feature extraction
                img = transform_img(read_image(image_path))
                with torch.no_grad():
                    pred_mask = seg_model(torch.from_numpy(img).float().unsqueeze(0))
                features = extract_morphometrics(pred_mask)

                # --- COMBINE METADATA AND FEATURES ---
                # Extract relevant metadata fields
                # Add all extracted features
                row = value.to_dict()
                row.update(features)

                # Append to DataFrame
                df_new_row = pd.DataFrame([row])
                df = pd.concat([df, df_new_row], ignore_index=True)

            except Exception as e:
                print(f"Error processing image ID {i} at index {i}: {e}. Skipping...")
                # Still save the progress before continuing to the next image
                df.to_csv(OUTPUT_PATH, index=False)
                continue

            # CHECKPOINTING LOGIC
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                
                print(f"\n--- Checkpointing progress at {i+1} images. ---\n")
                df.to_csv(OUTPUT_PATH, index=False)
        return df