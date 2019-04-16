import os
import pandas as pd
from core.live_data_processor import LiveDataLoader
import numpy as np
data = LiveDataLoader(
    filename=os.path.join('data', 'gld.csv'),
    split=0.8,
    cols=['Open', 'Close', 'Volume']
)


last_date = data.get_test_date(30)[-1]
df = pd.DataFrame({'DateTrain' : last_date})
print(df)
a = 1
