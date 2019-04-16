import fix_yahoo_finance as yf
from IPython.display import display
data = yf.download("GLD", '2019-3-1', '2019-4-15')
display(data)
print(len(data.index))