import pandas as pd

columns = [
    "party",
    "handicapped_infants",
    "water_project_cost_sharing",
    "adoption_of_the_budget_resolution",
    "physician_fee_freeze",
    "el_salvador_aid",
    "religious_groups_in_schools",
    "anti_satellite_test_ban",
    "aid_to_nicaraguan_contras",
    "mx_missile",
    "immigration",
    "synfuels_corporation_cutback",
    "education_spending",
    "superfund_right_to_sue",
    "crime",
    "duty_free_exports",
    "export_south_africa"
]

df = pd.read_csv(
    "house-votes-84.data",
    header=None,
    names=columns,
    skipinitialspace=True,
    keep_default_na=False
)

df.to_csv("voting_records_dirty.csv", index=False)

print("Question-mark missing values:")
print((df == "?").sum())

print("\nTotal ? values:")
print((df == "?").sum().sum())