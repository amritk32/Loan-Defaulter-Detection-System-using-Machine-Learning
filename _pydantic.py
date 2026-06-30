from pydantic import BaseModel, Field
from typing import Literal

class LoanApplication(BaseModel):
    annual_inc: int = Field(..., gt=0, description="Annual income of the applicant")
    loan_amnt: int = Field(..., gt=0, description="Total amount of the loan")
    int_rate: float = Field(..., ge=0, le=100, description="Interest rate as a percentage")
    emp_length: int = Field(..., ge=0, le=10, description="Years of employment (0-10)")
    dti: float = Field(..., ge=0, description="Debt-to-Income ratio")

    term: Literal[" 36 months", " 60 months"]

    home_ownership: Literal["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"]

    purpose: Literal[
        "debt_consolidation",
        "credit_card",
        "home_improvement",
        "other",
        "major_purchase",
        "small_business",
        "car",
        "medical",
        "wedding",
        "moving",
        "house",
        "vacation",
        "educational",
        "renewable_energy"
    ]

    total_acc: int = Field(..., ge=0, description="Total number of credit lines")
