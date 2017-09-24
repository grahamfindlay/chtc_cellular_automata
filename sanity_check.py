# Usage: python sanity_check.py relative/path/to/results/directory
import sys
import os
import glob
import pandas as pd
import pickle

dirname = sys.argv[-1] # directory containing pickles
files = glob.glob(os.path.join(os.path.abspath(dirname), '') + '*.pkl')

columns = ['upc_size', 'phi', 'mc_size']
df = pd.DataFrame(columns=columns)

for job in files:
    with open(job, 'rb') as f:
        mip = pickle.load(f)
        if mip:
            assert(mip.big_mip_past.unpartitioned_constellation ==
                   mip.big_mip_future.unpartitioned_constellation)
            data = [len(mip.big_mip_past.unpartitioned_constellation),
                    mip.phi,
                    len(mip.subsystem)]
        else:
            data = [None, None, None]
        row = pd.DataFrame([data], index=[os.path.basename(job)], columns=columns)
        df = df.append(row)

df.to_csv(os.path.join(os.path.abspath(dirname), 'sanity_check.csv'))
print(df)
print("Number of empty mips:", df.isnull().sum())
print("Summary statistics:\n", df.describe(include='all'))

