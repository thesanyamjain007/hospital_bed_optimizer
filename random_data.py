import pandas as pd
import numpy as np

np.random.seed(42)  # For reproducibility

n = 500

# Generate patient IDs
patient_ids = [f'P{str(i).zfill(5)}' for i in range(1, n+1)]

# Features: Simulate categorical and numerical columns
ages = np.random.randint(18, 95, size=n)
genders = np.random.choice(['Male', 'Female'], size=n)
admission_types = np.random.choice(['Emergency', 'Planned', 'Transfer'], size=n)
departments = np.random.choice(['Medicine', 'Surgery', 'Cardiology', 'Neurology', 'Orthopedics'], size=n)
diagnosis_categories = np.random.choice(['Infection', 'Cardiovascular', 'Respiratory', 'Trauma', 'Other'], size=n)
admission_hours = np.random.randint(0, 24, size=n)
days_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n)
severity_scores = np.random.randint(1, 16, size=n)
comorbidities_count = np.random.randint(0, 6, size=n)
previous_admissions = np.random.randint(0, 12, size=n)
icu_flag = np.random.choice([0, 1], size=n)
emergency_flag = np.random.choice([0, 1], size=n)
surgical_flag = np.random.choice([0, 1], size=n)
bed_types = np.random.choice(['ICU', 'General', 'Private'], size=n)
length_of_stay = np.random.randint(1, 21, size=n)
insurance_types = np.random.choice(['Public', 'Private', 'None'], size=n)
bed_occupancy_level = np.random.choice(['Low', 'Medium', 'High'], size=n)
discharge_within_24h = np.random.choice([0, 1], size=n)
estimated_stay_days = np.clip(length_of_stay + np.random.randint(-3, 4, size=n), 1, None)

# Target: Make 40% "long stay" (label=1) and 60% "short stay" (label=0)
long_stay_label = np.zeros(n, dtype=int)
long_stay_label[:int(0.4*n)] = 1  # 40% long stay
np.random.shuffle(long_stay_label)

# Compose DataFrame
df = pd.DataFrame({
    'patient_id': patient_ids,
    'age': ages,
    'gender': genders,
    'admission_type': admission_types,
    'department': departments,
    'diagnosis_category': diagnosis_categories,
    'admission_hour': admission_hours,
    'day_of_week': days_of_week,
    'severity_score': severity_scores,
    'comorbidities_count': comorbidities_count,
    'previous_admissions': previous_admissions,
    'icu_flag': icu_flag,
    'emergency_flag': emergency_flag,
    'surgical_flag': surgical_flag,
    'bed_type': bed_types,
    'length_of_stay': length_of_stay,
    'insurance_type': insurance_types,
    'long_stay_label': long_stay_label,
    'bed_occupancy_level': bed_occupancy_level,
    'discharge_within_24h': discharge_within_24h,
    'estimated_stay_days': estimated_stay_days
})

df.to_csv("hospital_bed_dataset_500.csv", index=False)
print("hospital_bed_dataset_500.csv generated successfully!")
