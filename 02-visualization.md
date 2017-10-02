# 2. Visualization and basic statistics

### 2.1. Averages and popularity

We first want to calculate some basic statistical data, such as the mean of the length of characters of all tweets, the tweet with more likes and retweets, etc.

From now, I'll just add some input code and the output right below the code.

To obtain the mean, using numpy:
```python
# We extract the mean of lenghts:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))
```
`The lenght's average in tweets: 125.925`

To extract more data, we will use some pandas' functionalities:
```python
# We extract the tweet with more FAVs and more RTs:

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))
```
<code>The tweet with more likes is: </code>
<code>The United States condemns the terror attack in Barcelona, Spain, and will do whatever is necessary to help. Be tough & strong, we love you!</code>
<code>Number of likes: 222205</code>
<code>144 characters.</code>

<code>The tweet with more retweets is: </code>
<code>The United States condemns the terror attack in Barcelona, Spain, and will do whatever is necessary to help. Be tough & strong, we love you!</code>
<code>Number of retweets: 66099</code>
<code>144 characters.</code>

This is common, but it won't necessarily happen: the tweet with more likes is the tweet with more retweets. What we're doing is that we find the maximum number of likes from the *'Likes'* column and the maximum number of retweets from the *'RTs'* using numpy's `max` function. With this we just look for the index in each of both columns that satisfy to be the maximum. Since more than one could have the same number of likes/retweets (the maximum) we just need to take the first one found, and that's why we use `.index[0]` to assign the index to the variables `fav`and `rt`. To print the tweet that satisfies, we access the data in the same way we would access a matrix or any indexed object.

We're now ready to plot some stuff. :)

### 2.2. Time series

Pandas has its own object for time series. Since we have a whole vector with creation dates, we can construct time series respect tweets lengths, likes and retweets.

The way we do it is:
```python
# We create time series for data:

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])
```

And if we want to plot the time series, pandas already has its own method in the object. We can plot a time series as follows:

```python
# Lenghts along time:
tlen.plot(figsize=(16,4), color='r');
```
This creates the following output:
![Lenghts along time](https://thepracticaldev.s3.amazonaws.com/i/jchdg3julb5qz86vqxlk.png)

And to plot the likes versus the retweets in the same chart:
```python
# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);
```
This will create the following output:
![Likes vs. retweets](https://thepracticaldev.s3.amazonaws.com/i/4xm53sj19anhyrc2fiyy.png)

### 2.3. Pie charts of sources

We're almost done with this second section of the workshop. Now we will plot the sources in a pie chart, since we realized that not every tweet is tweeted from the same source (😱🤔). We first clean all the sources:

```python
# We obtain all possible sources:
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

# We print sources list:
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))
```
With the following output, we realize that basically this twitter account has two sources:
<code>Creation of content sources:</code>
<code>* Twitter for iPhone</code>
<code>* Media Studio</code>

We now count the number of each source and create a pie chart. You'll notice that this code cell is not the most optimized one... Please have in mind that it was 4 in the morning when I was designing this workshop. 😅

```python
# We create a numpy vector mapped to labels:
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass

percent /= 100

# Pie chart:
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));
```

With this we obtain an output like this one:
![Pie chart of sources](https://thepracticaldev.s3.amazonaws.com/i/kkpiaea77x0r37lsotqs.png)

And we can see the percentage of tweets per source.

We can now proceed to do sentiment analysis.

> Go back to [1. Extracting twitter data (tweepy + pandas)](https://github.com/RodolfoFerro/pandas_twitter/blob/master/01-extracting-data.md)
> Go next to [3. Sentiment analysis](https://github.com/RodolfoFerro/pandas_twitter/blob/master/03-sentiment-analysis.md)
