"""
il/expert_profiles.py — Expert cleaning pipelines for Imitation Learning.

Three expert profiles are defined, one per dataset type:
  - BINARY    : categorical/binary data (e.g. voting records)
  - CONTINUOUS: numerical data with skewed distributions (e.g. adult income)
  - MEDICAL   : high-missingness correlated features (e.g. cancer datasets)

Each profile is a list of action indices (into the standard 8-action set)
representing the expert's recommended cleaning sequence.

Standard action set (order matters — indices used in SequentialCleaningEnvV3):
    0: MeanImputer
    1: MedianImputer
    2: KNNImputer
    3: IQROutlierCleaner
    4: ZScoreOutlierCleaner
    5: ExactDeduplicator
    6: MinMaxScaler
    7: ZScoreScaler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Action index constants — matches build_actions() in reproduce_table2.py
# ---------------------------------------------------------------------------
MEAN_IMPUTER      = 0
MEDIAN_IMPUTER    = 1
KNN_IMPUTER       = 2
IQR_OUTLIER       = 3
ZSCORE_OUTLIER    = 4
EXACT_DEDUP       = 5
MINMAX_SCALER     = 6
ZSCORE_SCALER     = 7


@dataclass
class ExpertProfile:
    """
    A named expert cleaning pipeline.

    Attributes
    ----------
    name : str
        Human-readable profile name.
    dataset_type : str
        One of "binary", "continuous", "medical".
    action_sequence : list[int]
        Ordered list of action indices the expert applies.
    description : str
        Rationale for this pipeline.
    """
    name: str
    dataset_type: str
    action_sequence: List[int]
    description: str = ""

    def __post_init__(self) -> None:
        valid_types = {"binary", "continuous", "medical", "high_dimensional", "dedup_heavy", "clean"}
        if self.dataset_type not in valid_types:
            raise ValueError(
                f"dataset_type must be one of {valid_types}, got {self.dataset_type!r}"
            )
        if not self.action_sequence:
            raise ValueError("action_sequence must not be empty.")


# ---------------------------------------------------------------------------
# Expert Profile Definitions
# ---------------------------------------------------------------------------

BINARY_EXPERT = ExpertProfile(
    name="BinaryExpert",
    dataset_type="binary",
    action_sequence=[
        MEAN_IMPUTER,    # Fill NaN with mean
        MEDIAN_IMPUTER,  # Also demonstrate median imputation
        EXACT_DEDUP,     # Remove duplicate rows
    ],
    description=(
        "For binary/categorical datasets (y/n votes, one-hot encoded features). "
        "Impute and deduplicate — never scale or apply IQR since all values "
        "are already in {0, 1} and outlier removal corrupts the binary encoding."
    ),
)

CONTINUOUS_EXPERT = ExpertProfile(
    name="ContinuousExpert",
    dataset_type="continuous",
    action_sequence=[
        MEAN_IMPUTER,    # Fill NaN with column mean
        KNN_IMPUTER,     # Fill remaining NaN with KNN (catches correlated patterns)
        IQR_OUTLIER,     # Clip extreme values
        EXACT_DEDUP,     # Remove duplicate rows
        # NOTE: Scaling removed — TabPFN v2 applies its own internal z-normalisation
        # and power scaling. Pre-scaling with MinMax/ZScore before TabPFN compounds
        # the transforms, degrading calibration (higher ECE) without improving accuracy.
    ],
    description=(
        "For continuous numerical datasets with skewed distributions (adult income, "
        "house prices). Impute → outlier removal → dedup. "
        "Scaling intentionally omitted — TabPFN handles normalisation internally."
    ),
)

MEDICAL_EXPERT = ExpertProfile(
    name="MedicalExpert",
    dataset_type="medical",
    action_sequence=[
        KNN_IMPUTER,     # KNN imputation — preserves feature correlations
        MEDIAN_IMPUTER,  # Also demonstrate median (robust to outliers in medical data)
        EXACT_DEDUP,     # Remove duplicate patient records
        IQR_OUTLIER,     # Remove physiological outliers
        # NOTE: ZScore scaling removed — same reason as ContinuousExpert.
        # TabPFN v2 handles normalisation internally.
    ],
    description=(
        "For medical/clinical datasets with high missingness and correlated features. "
        "KNN imputation preserves inter-feature correlations. No scaling — "
        "TabPFN handles normalisation internally to preserve calibration."
    ),
)


CLEAN_EXPERT = ExpertProfile(
    name="CleanExpert",
    dataset_type="clean",
    action_sequence=[
        IQR_OUTLIER,     # Handle skewed features (even clean datasets can have outliers)
        ZSCORE_SCALER,   # Z-score scaling for datasets with very different feature scales
    ],
    description=(
        "For datasets with no missing values and no duplicates. "
        "Focus only on outlier removal and scaling. "
        "Used when the data is already structurally clean."
    ),
)

DEDUP_HEAVY_EXPERT = ExpertProfile(
    name="DedupHeavyExpert",
    dataset_type="dedup_heavy",
    action_sequence=[
        EXACT_DEDUP,     # Fix duplicates first
        IQR_OUTLIER,     # Remove outliers with conservative factor
        MINMAX_SCALER,   # Scale — needed for skewed features like V2/V3 in blood_transfusion
    ],
    description=(
        "For datasets with high duplicate rates (>10%) and no/few missing values. "
        "Deduplication first, then conservative outlier removal, then scale. "
        "MinMax preferred over ZScore to preserve class distribution shape."
    ),
)

HIGH_DIMENSIONAL_EXPERT = ExpertProfile(
    name="HighDimensionalExpert",
    dataset_type="high_dimensional",
    action_sequence=[
        MEDIAN_IMPUTER,  # Median more robust than mean for high-dim sparse data
        EXACT_DEDUP,     # Remove duplicates
        IQR_OUTLIER,     # Remove outliers
        MINMAX_SCALER,   # Scale to [0,1] — needed for distance-based methods on wide data
    ],
    description=(
        "For high-dimensional datasets with many columns relative to rows. "
        "Median imputation (robust to outliers), dedup, outlier removal, then scale. "
        "MinMax scaling important for high-dim data where features have very different ranges."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_EXPERT_PROFILES: List[ExpertProfile] = [
    BINARY_EXPERT,
    CONTINUOUS_EXPERT,
    MEDICAL_EXPERT,
    HIGH_DIMENSIONAL_EXPERT,
    DEDUP_HEAVY_EXPERT,
    CLEAN_EXPERT,
]

EXPERT_PROFILES_BY_TYPE = {p.dataset_type: p for p in ALL_EXPERT_PROFILES}


def get_expert_profile(dataset_type: str) -> ExpertProfile:
    """
    Return the expert profile for a given dataset type.
    Falls back to CONTINUOUS_EXPERT for unknown types.
    """
    if dataset_type not in EXPERT_PROFILES_BY_TYPE:
        logger.warning(
            "No expert profile for dataset_type=%r — falling back to continuous.",
            dataset_type,
        )
        return EXPERT_PROFILES_BY_TYPE["continuous"]
    return EXPERT_PROFILES_BY_TYPE[dataset_type]


def describe_all() -> None:
    """Print a summary of all expert profiles."""
    for p in ALL_EXPERT_PROFILES:
        print(f"\n{'='*60}")
        print(f"  {p.name}  (type={p.dataset_type})")
        print(f"  Actions: {p.action_sequence}")
        print(f"  {p.description}")
