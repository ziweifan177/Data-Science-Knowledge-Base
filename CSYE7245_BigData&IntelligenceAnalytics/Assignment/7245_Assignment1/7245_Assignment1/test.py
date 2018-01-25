import pandas as pd
import numpy as np

dates=pd.date_range('20130101',periods=6)
df=pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['A','B','C','D'])
coreSamples=np.zeros_like(df,dtype=float)
print(coreSamples)

