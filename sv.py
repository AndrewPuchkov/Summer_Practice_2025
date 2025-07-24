import pandas as pd
import sweetviz as sv
df = pd.read_csv("quotes2.csv")
report = sv.analyze(df)
report.show_html('report.html')