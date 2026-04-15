
# WAND Framework

WAND (Wasserstein-based ANatomical Distance) is a general framework for morphometric evaluation consisting of:

1. Segmentation
2. Feature Extraction
3. Distribution Construction
4. Wasserstein Comparison

This repository provides an instantiation of WAND for chest radiography (WAND-CXR).

## Usage

```bash
python main.py \
  --csv_path chexpert_sample.csv \
  --num 10000 \
  --output_dir outputs/# CXR-WASABI
```