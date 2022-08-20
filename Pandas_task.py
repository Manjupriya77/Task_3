#1.Import pandas under the name pd.
import pandas as pd

#2. Print the version of pandas that has been imported.
print(pd.__version__)

#3.Print out all the version information of the libraries that are required by the pandas library.
print(pd.show_versions())


import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

#4.Create a DataFrame df from this dictionary data which has the index labels.
df=pd.DataFrame(data,index=labels)
print(df)

#5.Display a summary of the basic information about this DataFrame and its data.
df.info()

#6.Return the first 3 rows of the DataFrame df.
print(df.head(3))

#7.Select just the 'animal' and 'age' columns from the DataFrame df.
print(df[['animal', 'age']])

#8.Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].
print(df.loc[df.index[[3, 4, 8]], ['animal', 'age']])

#9.Select only the rows where the number of visits is greater than 3.
print(df[df['visits'] > 3])

#10.Select the rows where the age is missing, i.e. is NaN.
df[df['age'].isnull()]

#11.Select the rows where the animal is a cat and the age is less than 3.
df[(df['animal'] == 'cat') & (df['age'] < 3)]


#12.Select the rows the age is between 2 and 4 (inclusive).
df[df['age'].between(2, 4)]

#13.Change the age in row 'f' to 1.5.
df.loc['f', 'age'] = 1.5

#14.Calculate the sum of all visits (the total number of visits).
df['visits'].sum()

#15.Calculate the mean age for each different animal in df.
df.groupby('animal')['age'].mean()

#16.Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.
df.loc['k'] = [5.5, 'dog', 'no', 2]
df = df.drop('k')

#17.Count the number of each type of animal in df.
df['animal'].value_counts()

#18.Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending order.
df.sort_values(by=['age', 'visits'], ascending=[False, True])

#19.The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.
df['priority'] = df['priority'].map({'yes': True, 'no': False})

#20.In the 'animal' column, change the 'snake' entries to 'python'.
df['animal'] = df['animal'].replace('snake', 'python')

#21.For each animal type and each number of visits, find the mean age. In other words, each row is an animal,
#each column is a number of visits and the values are the mean ages (hint: use a pivot table).
df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')


#22.You have a DataFrame df with a column 'A' of integers. For example:
#How do you filter out rows which contain the same integer as the row immediately above?
df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
df.loc[df['A'].shift() != df['A']]

#23.Given a DataFrame of numeric values, say
#how do you subtract the row mean from each element in the row?
df = pd.DataFrame(np.random.random(size=(5, 3)))
df.sub(df.mean(axis=1), axis=0)

#24.Suppose you have DataFrame with 10 columns of real numbers, for example:
#Which column of numbers has the smallest sum? (Find that column's label.)
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
df.sum().idxmin()

#25.ow do you count how many unique rows a DataFrame has (i.e. ignore all rows that are duplicates)?
len(df) - df.duplicated(keep=False).sum()

#26.You have a DataFrame that consists of 10 columns of floating--point numbers. Suppose that exactly 5 entries in each row are NaN values.
#For each row of the DataFrame, find the column which contains the third NaN value.
(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1)

#27. A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example:
#For each group, find the sum of the three greatest values.
df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})

df.groupby('grp')['vals'].nlargest(3).sum(level=0)

#28.A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive).
#For each group of 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...), calculate the sum of the corresponding values in column 'B'.
df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()

#29. Consider a DataFrame df where there is an integer column 'X':
#For each value, count the difference back to the previous zero (or the start of the Series, whichever is closer).
#These values should therefore be [1, 2, 0, 1, 2, 3, 4, 0, 1, 2]. Make this a new column 'Y'.
df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
izero = np.r_[-1, (df['X'] == 0).nonzero()[0]] # indices of zeros
idx = np.arange(len(df))
df['Y'] = idx - izero[np.searchsorted(izero - 1, idx) - 1]


#30.Consider a DataFrame containing rows and columns of purely numerical data. Create a list of the row-column index locations of the 3 largest values
df.unstack().sort_values()[-3:].index.tolist()

#31. Given a DataFrame with a column of group IDs, 'grps', 
#and a column of corresponding integer values, 'vals', replace any negative values in 'vals' with the group mean.
def replace(group):
    mask = group<0
    group[mask] = group[~mask].mean()
    return group
df.groupby(['grps'])['vals'].transform(replace)

#32.Implement a rolling mean over groups with window size 3, which ignores NaN value. For example consider the following DataFrame:

df = pd.DataFrame({'group': list('aabbabbbabab'),
                       'value': [1, 2, 3, np.nan, 2, 3, 
                                 np.nan, 1, 7, 3, np.nan, 8]})
df
   group  value
0      a    1.0
1      a    2.0
2      b    3.0
3      b    NaN
4      a    2.0
5      b    3.0
6      b    NaN
7      b    1.0
8      a    7.0
9      b    3.0
10     a    NaN
11     b    8.0
#The goal is to compute the Series:
#0 1.000000 1 1.500000 2 3.000000 3 3.000000 4 1.666667 5 3.000000 6 3.000000 7 2.000000 8 3.666667 9 2.000000 10 4.500000 11 4.000000

g1 = df.groupby(['group'])['value']             
g2 = df.fillna(0).groupby(['group'])['value']    
s = g2.rolling(3, min_periods=1).sum() / g1.rolling(3, min_periods=1).count()
s.reset_index(level=0, drop=True).sort_index()

#33.Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers. Let's call this Series s.
dti = pd.date_range(start='2015-01-01', end='2015-12-31', freq='B') 
s = pd.Series(np.random.rand(len(dti)), index=dti)

#34.Find the sum of the values in s for every Wednesday.
s[s.index.weekday == 2].sum()

#35. For each calendar month in s, find the mean of values.
s.resample('M').mean()

#36.For each group of four consecutive calendar months in s, find the date on which the highest value occurred.
s.groupby(pd.TimeGrouper('4M')).idxmax()

#37.Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.
pd.date_range('2015-01-01', '2016-12-31', freq='WOM-3THU')


df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
#38.Some values in the the FlightNumber column are missing. These numbers are meant to increase by 10 with each row so 10055 and 10075 need to be put in place.
#Fill in these missing numbers and make the column an integer column (instead of a float column).
df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

#39.The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ to give a new temporary DataFrame with the correct values. 
#Assign the correct column names to this temporary DataFrame.
temp = df.From_To.str.split('_', expand=True)
temp.columns = ['From', 'To']

#40.Notice how the capitalisation of the city names is all mixed up in this temporary DataFrame.
#Standardise the strings so that only the first letter is uppercase (e.g. "londON" should become "London".)

temp['From'] = temp['From'].str.capitalize()
temp['To'] = temp['To'].str.capitalize()

#41. Delete the From_To column from df and attach the temporary DataFrame from the previous questions.
df = df.drop('From_To', axis=1)
df = df.join(temp)

#42. In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names.
#Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()

#43. In the RecentDelays column, the values have been entered into the DataFrame as a list.
#We would like each first value in its own column, each second value in its own column, and so on. If there isn't an Nth value, the value should be NaN.
#Expand the Series of lists into a DataFrame named delays, rename the columns delay_1, delay_2, etc. and replace the unwanted RecentDelays column in df with delays.
delays = df['RecentDelays'].apply(pd.Series)

delays.columns = ['delay_{}'.format(n) for n in range(1, len(delays.columns)+1)]

df = df.drop('RecentDelays', axis=1).join(delays)

#44.Given the lists letters = ['A', 'B', 'C'] and numbers = list(range(10)), construct a MultiIndex object from the product of the two lists.
#Use it to index a Series of random numbers. Call this Series s.
letters = ['A', 'B', 'C']
numbers = list(range(10))

mi = pd.MultiIndex.from_product([letters, numbers])
s = pd.Series(np.random.rand(30), index=mi)

#45.Check the index of s is lexicographically sorted (this is a necessary proprty for indexing to work correctly with a MultiIndex).
s.index.is_lexsorted()

#46.Select the labels 1, 3 and 6 from the second level of the MultiIndexed Series.

s.loc[:, [1, 3, 6]]

#47.Slice the Series s; slice up to label 'B' for the first level and from label 5 onwards for the second level.
s.loc[pd.IndexSlice[:'B', 5:]]
s.loc[slice(None, 'B'), slice(5, None)]

#48.Sum the values in s for each label in the first level (you should have Series giving you a total for labels A, B and C).
s.sum(level=0)

#49.Suppose that sum() (and other methods) did not accept a level keyword argument. How else could you perform the equivalent of s.sum(level=1)?
s.unstack().sum(axis=0)

#50. Exchange the levels of the MultiIndex so we have an index of the form (letters, numbers). Is this new Series properly lexsorted? If not, sort it.
new_s = s.swaplevel(0, 1)
new_s.index.is_lexsorted()
new_s = new_s.sort_index()

#51.Let's suppose we're playing Minesweeper on a 5 by 4 grid, i.e.

p = pd.tools.util.cartesian_product([np.arange(X), np.arange(Y)])
df = pd.DataFrame(np.asarray(p).T, columns=['x', 'y'])

#52. For this DataFrame df, create a new column of zeros (safe) and ones (mine). The probability of a mine occuring at each location should be 0.4.
df['mine'] = np.random.binomial(1, 0.4, X*Y)

#53.. Now create a new column for this DataFrame called 'adjacent'. This column should contain the number of mines found on adjacent squares in the grid.
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
        
from scipy.signal import convolve2d

mine_grid = df.pivot_table(columns='x', index='y', values='mine')
counts = convolve2d(mine_grid.astype(complex), np.ones((3, 3)), mode='same').real.astype(int)
df['adjacent'] = (counts - mine_grid).ravel('F')

#54. For rows of the DataFrame that contain a mine, set the value in the 'adjacent' column to NaN.
df.loc[df['mine'] == 1, 'adjacent'] = np.nan

#55.Finally, convert the DataFrame to grid of the adjacent mine counts: columns are the x coordinate, rows are the y coordinate.
df.drop('mine', axis=1)\
  .set_index(['y', 'x']).unstack()

'''56. Pandas is highly integrated with the plotting library matplotlib, and makes plotting DataFrames very user-friendly! Plotting in a notebook environment usually makes use of the following boilerplate:

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
matplotlib is the plotting library which pandas' plotting functionality is built upon, and it is usually aliased to plt.

%matplotlib inline tells the notebook to show plots inline, instead of creating them in a separate window.

plt.style.use('ggplot') is a style theme that most people find agreeable, based upon the styling of R's ggplot package.

For starters, make a scatter plot of this random data, but use black X's instead of the default markers.

df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})'''

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})

df.plot.scatter("xs", "ys", color = "black", marker = "x")



'''57.Columns in your DataFrame can also be used to modify colors and sizes. Bill has been keeping track of his performance at work over time, as well as how good he was feeling that day, and whether he had a cup of coffee in the morning. Make a plot which incorporates all four features of this DataFrame.

(Hint: If you're having trouble seeing the plot, try multiplying the Series which you choose to represent size by 10 or more)

The chart doesn't have to be pretty: this isn't a course in data viz!'''
df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})

df.plot.scatter("hours_in", "productivity", s = df.happiness * 30, c = df.caffienated)
'''58.What if we want to plot multiple things? Pandas allows you to pass in a matplotlib Axis object for plots, and plots will also return an Axis object.

Make a bar plot of monthly revenue with a line plot of monthly advertising spending (numbers in millions)'''
df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })

ax = df.plot.bar("month", "revenue", color = "green")
df.plot.line("month", "advertising", secondary_y = True, ax = ax)
ax.set_xlim((-1,12))
df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })

ax = df.plot.bar("month", "revenue", color = "green")
df.plot.line("month", "advertising", secondary_y = True, ax = ax)
ax.set_xlim((-1,12))

'''59.Generate a day's worth of random stock data, and aggregate / reformat it so that it has hourly summaries of the opening, highest, 
lowest, and closing prices'''
df = day_stock_data()
df.head()
df.set_index("time", inplace = True)
agg = df.resample("H").ohlc()
agg.columns = agg.columns.droplevel()
agg["color"] = (agg.close > agg.open).map({True:"green",False:"red"})
agg.head()

'''60.Now that you have your properly-formatted data, try to plot it yourself as a candlestick chart.
Use the plot_candlestick(df) function above, or matplotlib's plot documentation if you get stuck.'''
plot_candlestick(agg)
