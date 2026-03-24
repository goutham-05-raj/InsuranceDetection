import pandas as pd
import numpy as np
import json
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger
from src.utils.exceptions import PreprocessingError

logger = get_logger(__name__)

def extract_features(claim: dict) -> dict:
    """Extract numeric features required for modeling."""
    try:
        features = {
            "TotalClaimed": claim["ClaimAmounts"]["TotalClaimed"],
            "TotalApproved": claim["ClaimAmounts"]["TotalApproved"],
            "CreditScore": claim["ClaimantFinancialInformation"]["CreditScore"],
            "AnnualIncome": claim["ClaimantFinancialInformation"]["AnnualIncome"],
            "DebtToIncomeRatio": claim["ClaimantFinancialInformation"]["DebtToIncomeRatio"],
            "ClaimFrequency": claim["ClaimantBehavior"]["ClaimFrequency"],
            "LatePayments": claim["ClaimantBehavior"]["LatePayments"],
            "PolicyChanges": claim["ClaimantBehavior"]["PolicyChanges"],
            "CoverageBIL": claim["Coverage"]["BIL"]["ClaimedAmount"],
            "CoveragePDL": claim["Coverage"]["PDL"]["ClaimedAmount"],
            "CoveragePIP": claim["Coverage"]["PIP"]["ClaimedAmount"],
            "CoverageCollision": claim["Coverage"]["CollisionCoverage"]["ClaimedAmount"],
            "CoverageComprehensive": claim["Coverage"]["ComprehensiveCoverage"]["ClaimedAmount"],
            # Feature engineering additions:
            "ClaimedToIncomeRatio": claim["ClaimAmounts"]["TotalClaimed"] / max(claim["ClaimantFinancialInformation"]["AnnualIncome"], 1),
            "ApprovedToClaimedRatio": claim["ClaimAmounts"]["TotalApproved"] / max(claim["ClaimAmounts"]["TotalClaimed"], 1)
        }
        return features
    except KeyError as e:
        logger.error(f"Missing key during feature extraction: {e}")
        raise PreprocessingError(f"Feature extraction failed on {e}")

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """Load JSON claims and construct a feature DataFrame."""
    logger.info(f"Loading data from {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    
    feature_list = []
    labels = []
    
    for claim in data:
        features = extract_features(claim)
        label = 1 if claim.get("is_abnormal", False) else 0
        feature_list.append(features)
        labels.append(label)
        
    df = pd.DataFrame(feature_list)
    df["target"] = labels
    logger.info(f"Preprocessed DataFrame shape: {df.shape}")
    return df

def apply_smote(X: pd.DataFrame, y: pd.Series):
    """Apply SMOTE to balance class distribution."""
    logger.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(f"Target distribution after SMOTE:\n{pd.Series(y_res).value_counts()}")
    return X_res, y_res
