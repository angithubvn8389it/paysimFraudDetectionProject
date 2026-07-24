from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("data\\raw_data\\paysimLog.csv")

profile = ProfileReport(df, title="Dataset Overview")
profile.to_file("report.html")