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
        valid_types = {"binary", "continuous", "medical"}
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
        MEAN_IMPUTER,   # Fill NaN with mean (0/1 binary → fills with modal-ish value)
        EXACT_DEDUP,    # Remove duplicate rows (voting records have exact duplicates)
    ],
    description=(
        "For binary/categorical datasets (y/n votes, one-hot encoded features). "
        "Only impute and deduplicate — never scale or apply IQR since all values "
        "are already in {0, 1} and outlier removal corrupts the binary encoding."
    ),
)

CONTINUOUS_EXPERT = ExpertProfile(
    name="ContinuousExpert",
    dataset_type="continuous",
    action_sequence=[
        MEAN_IMPUTER,   # Fill NaN with column mean
        IQR_OUTLIER,    # Clip extreme values (adult: capital_gain, fnlwgt are skewed)
        EXACT_DEDUP,    # Remove duplicate rows
        MINMAX_SCALER,  # Scale to [0,1] for consistent feature magnitudes
    ],
    description=(
        "For continuous numerical datasets with skewed distributions (adult income, "
        "house prices). Mean impute first, then clip outliers with IQR, deduplicate, "
        "and scale. IQR uses factor 3.0 to avoid over-aggressive removal on skewed cols."
    ),
)

MEDICAL_EXPERT = ExpertProfile(
    name="MedicalExpert",
    dataset_type="medical",
    action_sequence=[
        KNN_IMPUTER,    # KNN imputation — preserves feature correlations (medical data is correlated)
        EXACT_DEDUP,    # Remove duplicate patient records
        ZSCORE_SCALER,  # Z-score normalisation — medical features have very different scales
    ],
    description=(
        "For medical/clinical datasets with high missingness and correlated features "
        "(cancer, diabetes, hepatitis). KNN imputation preserves inter-feature correlations "
        "that are clinically meaningful. Z-score scaling handles the extreme scale differences "
        "between features like blood pressure (60-200) and glucose (0.5-5.0)."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_EXPERT_PROFILES: List[ExpertProfile] = [
    BINARY_EXPERT,
    CONTINUOUS_EXPERT,
    MEDICAL_EXPERT,
]

EXPERT_PROFILES_BY_TYPE = {p.dataset_type: p for p in ALL_EXPERT_PROFILES}


def get_expert_profile(dataset_type: str) -> ExpertProfile:
    """
    Return the expert profile for a given dataset type.

    Parameters
    ----------
    dataset_type : str
        One of "binary", "continuous", "medical".

    Returns
    -------
    ExpertProfile

    Raises
    ------
    KeyError if dataset_type is not registered.
    """
    if dataset_type not in EXPERT_PROFILES_BY_TYPE:
        raise KeyError(
            f"No expert profile for dataset_type={dataset_type!r}. "
            f"Available: {list(EXPERT_PROFILES_BY_TYPE.keys())}"
        )
    return EXPERT_PROFILES_BY_TYPE[dataset_type]


def describe_all() -> None:
    """Print a summary of all expert profiles."""
    for p in ALL_EXPERT_PROFILES:
        print(f"\n{'='*60}")
        print(f"  {p.name}  (type={p.dataset_type})")
        print(f"  Actions: {p.action_sequence}")
        print(f"  {p.description}")
