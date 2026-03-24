from pydantic import BaseModel, Field
from typing import List

class ClaimFeatures(BaseModel):
    # Base amounts
    TotalClaimed: float = Field(..., description="Total amount claimed")
    TotalApproved: float = Field(..., description="Total amount approved")
    
    # Financial metrics
    CreditScore: int = Field(..., description="Claimant Credit Score")
    AnnualIncome: float = Field(..., description="Claimant Annual Income")
    DebtToIncomeRatio: float = Field(..., description="Debt to Income Ratio")
    
    # Behavioral metrics
    ClaimFrequency: int = Field(..., description="Number of past claims")
    LatePayments: int = Field(..., description="Number of late payments")
    PolicyChanges: int = Field(..., description="Number of policy changes")
    
    # Coverages
    CoverageBIL: float = Field(..., description="Bodily Injury Liability Claimed Amount")
    CoveragePDL: float = Field(..., description="Property Damage Liability Claimed Amount")
    CoveragePIP: float = Field(..., description="Personal Injury Protection Claimed Amount")
    CoverageCollision: float = Field(..., description="Collision Coverage Claimed Amount")
    CoverageComprehensive: float = Field(..., description="Comprehensive Coverage Claimed Amount")
    
    # Engineer Features
    ClaimedToIncomeRatio: float = Field(..., description="TotalClaimed / AnnualIncome")
    ApprovedToClaimedRatio: float = Field(..., description="TotalApproved / TotalClaimed")
    
class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_prediction: int
    risk_level: str
    
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
