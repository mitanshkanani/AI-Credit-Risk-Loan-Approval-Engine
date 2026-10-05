import pandas as pd
import sweetviz as sv

print("Loading accepted dataset...")
accepted = pd.read_csv("data/accepted_2007_to_2018Q4.csv")

print("Generating accepted report...")
accepted["id"] = accepted["id"].astype(str)
accepted_report = sv.analyze(accepted)
accepted_report.show_html("accepted_dataset_report.html")

print("Accepted report complete.")

print("Loading rejected dataset...")
rejected = pd.read_csv("data/rejected_2007_to_2018Q4.csv")

print("Generating rejected report...")
rejected_report = sv.analyze(rejected)
rejected_report.show_html("rejected_dataset_report.html")

print("Rejected report complete.")
print("Both reports generated successfully.")