#%%
import os
import glob

import pandas as pd
import xarray as xr
# %%
data_dir = "../data/swath/"
os.makedirs(data_dir, exist_ok=True)
exist_data = list(map(os.path.basename, glob.glob(f"{data_dir}/*")))
# %%
request_url = "https://cyclobs.ifremer.fr/app/api/getData?instrument=C-Band_SAR&include_cols=all"
data_list = pd.read_csv(request_url)
eye_list = data_list[data_list["eye_in_acq"]==True]
# %%
for eye_data_url in eye_list["data_url"]:
    if os.path.basename(eye_data_url) not in exist_data:
        ret = os.system(f"cd {data_dir}; wget -N  {eye_data_url}")
# %%
