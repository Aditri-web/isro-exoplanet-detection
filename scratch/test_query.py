from astroquery.mast import Observations
import pandas as pd

print("Querying TESS Sector 1 observations...")
obs = Observations.query_criteria(
    project="TESS",
    sequence_number=1,
    obs_collection="TESS"
)
df = obs.to_pandas()
print(list(df.columns))
print(df[["target_name", "s_ra", "s_dec"]].head(10))
