import pickle
from model import normalize_record

with open('pipeline.bin', 'rb') as f:
    p = pickle.load(f)

a = {"lead_source": "paid_ads", "industry": "technology", "employment_status": "employed",
     "location": "north_america", "number_of_courses_viewed": 2, "annual_income": 79276.0,
     "interaction_count": 4, "lead_score": 0.41}
b = {"lead_source": "organic_search", "industry": "technology", "employment_status": "employed",
     "location": "europe", "number_of_courses_viewed": 4, "annual_income": 80304.0,
     "interaction_count": 7, "lead_score": 0.74}

print("Q3", round(p.predict_proba([normalize_record(a)])[0, 1], 3))
print("Q4", round(p.predict_proba([normalize_record(b)])[0, 1], 3))