import pandas as pd
def quality_control(anatomical_measuements_df:pd.DataFrame):
    """
    Perform quality control on the anatomical measurements DataFrame.
    - Simple Heuristic: Remove rows with any feature outside the range [0, 1] (normalized area measurements).
    - failed segmentation can lead to zero or very high values, which are likely QC failures.
    
    Returns:
    - qc_passed_df: DataFrame containing only the rows that passed quality control.
    """
    columsn_to_drop = [
    "id",
    "prompt",
    "path",
    "report",
    "class_label",
    "Thorax_Width",
    "Spine_Length",
    "Thoracic_Ref_Area",
    ]
    anatomical_measuements_df = anatomical_measuements_df.drop(columns=columsn_to_drop, errors='ignore')

    # Define QC criteria (these are examples and should be adjusted based on domain knowledge)
    qc_passed_df = anatomical_measuements_df.copy()
    min_val = 0.
    max_val = 1.0
    # any feature outside the range [min_val, max_val] is considered a QC failure
    qc_passed_df = qc_passed_df[
        (qc_passed_df > min_val).all(axis=1) & (qc_passed_df <= max_val).all(axis=1)
    ]
    
    return qc_passed_df