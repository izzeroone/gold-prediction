import fix_yahoo_finance as yf
data = yf.download('GLD','2000-01-01')
data.to_csv('data/gld.csv')