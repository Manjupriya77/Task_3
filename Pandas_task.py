import pandas as pd

print(pd.__version__)

print(pd.show_versions())
import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df=pd.DataFrame(data,index=labels)
print(df)
print()

df.info()
print()
print(df.head(3))

print()
print(df[['animal', 'age']])

print()
print(df.loc[df.index[[3, 4, 8]], ['animal', 'age']])

print()
print(df[df['visits'] > 3])

df[df['age'].isnull()]

df[(df['animal'] == 'cat') & (df['age'] < 3)]

df[df['age'].between(2, 4)]

df.loc['f', 'age'] = 1.5

df['visits'].sum()

df.groupby('animal')['age'].mean()

df.loc['k'] = [5.5, 'dog', 'no', 2]

# and then deleting the new row...

df = df.drop('k')


df.loc[df['A'].shift() != df['A']]

df.sub(df.mean(axis=1), axis=0)

df.sum().idxmin()

len(df) - df.duplicated(keep=False).sum()

(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)

df.groupby('grp')['vals'].nlargest(3).sum(level=0)

df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()

izero = np.r_[-1, (df['X'] == 0).nonzero()[0]] # indices of zeros
idx = np.arange(len(df))
df['Y'] = idx - izero[np.searchsorted(izero - 1, idx) - 1]


x = (df['X'] != 0).cumsum()
y = x != x.shift()
df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()


df['Y'] = df.groupby((df['X'] == 0).cumsum()).cumcount()
# We're off by one before we reach the first zero.
first_zero_idx = (df['X'] == 0).idxmax()
df['Y'].iloc[0:first_zero_idx] += 1


df.unstack().sort_values()[-3:].index.tolist()

def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group

df.groupby(['grps'])['vals'].transform(replace)


g1 = df.groupby(['group'])['value']              # group values  
g2 = df.fillna(0).groupby(['group'])['value']    # fillna, then group values

s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count() # compute means

s.reset_index(level=0, drop=True).sort_index()  # drop/sort index


dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)


s[s.index.weekday == 2].sum()


s.resample('M').mean()

s.groupby(pd.TimeGrouper('4M')).idxmax()

pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')


df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']


temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()


df = df.drop('From_To', axis=1)
df = df.join(temp)
df['animal'].value_counts()

df.sort_values(by=['age', 'visits'], ascending=[False, True])

df['priority'] = df['priority'].map({'yes': True, 'no': False})

df['animal'] = df['animal'].replace('snake', 'python')

df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')


df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()


delays = df['RecentDelays'].apply(pd.Series)

delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]

df = df.drop('RecentDelays', axis=1).join(delays)


letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)

s.index.is_lexsorted()


s.loc[:, [1, 3, 6]]

s.loc[pd.IndexSlice[:'B', 5:]]

s.sum(level=0)

s.unstack().sum(axis=0)

new_s = s.swaplevel(0, 1)

# check
new_s.index.is_lexsorted()

# sort
new_s = new_s.sort_index()

p = pd.tools.util.cartesian_product([np.arange(X), np.arange(Y)])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])


df['mine'] = np.random.binomial(1, 0.4, X*Y)


df['adjacent'] = \
    df.merge(df + [ 1,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 1, -1, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1, -1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 1,  0, 0], on=['x', 'y'], how='left')\
      .merge(df + [-1,  0, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 0,  1, 0], on=['x', 'y'], how='left')\
      .merge(df + [ 0, -1, 0], on=['x', 'y'], how='left')\
       .iloc[:, 3:]\
        .sum(axis=1)
        
# An alternative solution is to pivot the DataFrame 
# to form the "actual" grid of mines and use convolution.
# See https://github.com/jakevdp/matplotlib_pydata2013/blob/master/examples/minesweeper.py

from scipy.signal import convolve2d

mine_grid = df.pivot_table(columns='x', index='y', values='mine')
counts = convolve2d(mine_grid.astype(complex), np.ones((3, 3)), mode='same').real.astype(int)
df['adjacent'] = (counts - mine_grid).ravel('F')


df.loc[df['mine'] == 1, 'adjacent'] = np.nan

df.drop('mine', axis=1)\
  .set_index(['y', 'x']).unstack()


import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})

df.plot.scatter("xs", "ys", color = "black", marker = "x")



        ax.plot([time.hour] * 2, agg.loc[time, ["open","close"]].values, color = agg.loc[time, "color"], linewidth = 10)

    ax.set_xlim((8,16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()
    
    
    df = day_stock_data()
df.head()

df.set_index("time", inplace = True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True:"green",False:"red"})
agg.head()


plot_candlestick(agg)
