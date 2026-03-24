import json
import random
import uuid
import os
from datetime import datetime, timedelta
from faker import Faker
from src.utils.logger import get_logger
from src.utils.exceptions import DataGenerationError
import yaml

fake = Faker()
logger = get_logger(__name__)

def load_schema(schema_path: str):
    try:
        with open(schema_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        raise DataGenerationError(f"Schema loading error: {e}")

def generate_policy_holder(policy_holder_id, schema):
    return {
        "PolicyHolderID": str(uuid.uuid4()),
        "Name": fake.name(),
        "PolicyNumber": str(uuid.uuid4()),
        "ContactInformation": {
            "Phone": fake.phone_number(),
            "Email": fake.email(),
            "Address": fake.address()
        }
    }

def generate_claim(policy_holder, claim_id, schema, is_abnormal=False):
    start_date = datetime.now() - timedelta(days=random.randint(0, 365))
    end_date = start_date + timedelta(days=365)
    accident_date = start_date + timedelta(days=random.randint(0, 365))
    
    coverage_limits = [10000, 20000, 50000, 100000, 200000]
    
    def random_claimed_amount(limit):
        return random.randint(1000, limit)
    
    claim = {
        "ClaimID": claim_id,
        "PolicyID": policy_holder["PolicyNumber"],
        "EffectiveDates": {
            "StartDate": start_date.strftime("%Y-%m-%d"),
            "EndDate": end_date.strftime("%Y-%m-%d")
        },
        "PolicyHolder": policy_holder,
        "AccidentDetails": {
            "Date": accident_date.strftime("%Y-%m-%d"),
            "Location": fake.address(),
            "Description": fake.text(max_nb_chars=200)
        },
        "Coverage": {
            "BIL": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "PDL": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "PIP": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "CollisionCoverage": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            },
            "ComprehensiveCoverage": {
                "CoverageLimit": random.choice(coverage_limits),
                "ClaimedAmount": random_claimed_amount(random.choice(coverage_limits))
            }
        },
        "ClaimStatus": random.choice(["Filed", "In Review", "Approved", "Closed"]),
        "ClaimAmounts": {
            "TotalClaimed": 0,
            "TotalApproved": 0
        },
        "AdjusterDetails": {
            "Name": fake.name(),
            "ContactInformation": {
                "Phone": fake.phone_number(),
                "Email": fake.email()
            }
        },
        "SupportingDocuments": [
            {
                "DocumentType": random.choice(["Police Report", "Medical Report", "Repair Estimate", "Witness Statement"]),
                "DocumentURL": f"http://example.com/documents/{uuid.uuid4()}"
            }
        ],
        "ClaimHistory": [
            {
                "Status": "Filed",
                "Date": accident_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "Notes": "Initial claim filed"
            }
        ],
        "VehicleInformation": {
            "Make": fake.company(),
            "Model": fake.word(),
            "Year": random.randint(2000, 2022),
            "VIN": str(uuid.uuid4())
        },
        "WitnessInformation": [
            {
                "Name": fake.name(),
                "ContactInformation": {
                    "Phone": fake.phone_number(),
                    "Email": fake.email()
                },
                "Statement": fake.text(max_nb_chars=200)
            }
        ],
        "PoliceReport": {
            "ReportID": str(uuid.uuid4()),
            "OfficerName": fake.name(),
            "ReportDetails": fake.text(max_nb_chars=200)
        },
        "ClaimantFinancialInformation": {
            "CreditScore": random.randint(300, 850),
            "AnnualIncome": random.randint(20000, 200000),
            "DebtToIncomeRatio": round(random.uniform(0.1, 1.0), 2)
        },
        "ClaimantBehavior": {
            "ClaimFrequency": random.randint(0, 10),
            "LatePayments": random.randint(0, 10),
            "PolicyChanges": random.randint(0, 10)
        },
        "is_abnormal": is_abnormal
    }
    
    claim["ClaimAmounts"]["TotalClaimed"] = sum([coverage["ClaimedAmount"] for coverage in claim["Coverage"].values()])
    claim["ClaimAmounts"]["TotalApproved"] = random.randint(1000, claim["ClaimAmounts"]["TotalClaimed"])
    
    if is_abnormal:
        claim["ClaimantFinancialInformation"]["CreditScore"] = random.randint(300, 500)
        claim["ClaimantBehavior"]["ClaimFrequency"] = random.randint(5, 10)
        claim["ClaimantBehavior"]["LatePayments"] = random.randint(5, 10)
        claim["ClaimantBehavior"]["PolicyChanges"] = random.randint(5, 10)
        for coverage in claim["Coverage"].values():
            coverage["ClaimedAmount"] = random.randint(4000000, 5000000)
        claim["ClaimAmounts"]["TotalClaimed"] = sum([coverage["ClaimedAmount"] for coverage in claim["Coverage"].values()])
        claim["ClaimAmounts"]["TotalApproved"] = random.randint(1000000, claim["ClaimAmounts"]["TotalClaimed"])
    
    return claim

def generate_claims(num_claims, num_policy_holders, schema, abnormal=False):
    logger.info(f"Generating {num_claims} claims... abnormal={abnormal}")
    policy_holders = [generate_policy_holder(i, schema) for i in range(num_policy_holders)]
    claims = []
    
    # Let's say 15% fraud rate if normal, otherwise mostly fraud if abnormal flag is on
    fraud_rate = 0.5 if abnormal else 0.15
    num_fraudulent_claims = int(num_claims * fraud_rate)
    
    for i in range(num_claims):
        policy_holder = random.choice(policy_holders)
        is_fraudulent = i < num_fraudulent_claims
        claim = generate_claim(policy_holder, str(uuid.uuid4()), schema, is_fraudulent)
        claims.append(claim)
    
    logger.info("Claims generation completed.")
    return claims

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    schema = load_schema(config["data"]["schema_path"])
    claims = generate_claims(1000, 100, schema)
    # Save to list
    os.makedirs(config["data"]["raw_path"], exist_ok=True)
    with open(f"{config['data']['raw_path']}/claims.json", "w") as f:
        json.dump(claims, f, indent=4)
