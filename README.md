# Sentiment analysis on Trump's tweets (workshop) using Python 🐍

> **DESCRIPTION:**
>In this workshop we will:
>1. Extract twitter data using tweepy and learn how to handle it using pandas.
>2. Do some basic statistics and visualizations with numpy, matplotlib and seaborn.
>3. Do sentiment analysis of extracted (Trump's) tweets using textblob.


I hope you find this a bit useful and/or interesting. I taught this workshop in *Mexico City*. I'll explain the whole instructions along with code, in the most simple way possible. Anyway, all the code can be found in the [Jupyter Notebook I used for this workshop](https://github.com/RodolfoFerro/pandas_twitter/blob/master/Minado%20de%20datos%20en%20Twitter.ipynb).

# What will we need?

First of all, we need to have [Python](https://www.python.org/downloads/) installed.
I'm almost sure that all the code will run in Python 2.7, but I'll use Python 3.6. I highly recommend to install [Anaconda](https://www.anaconda.com/download/), which is a very useful Python distribution to manage packages that includes a lot of useful tools, such as [Jupyter Notebooks](http://jupyter.org/). I'll explain the code supposing that we will be using a Jupyter Notebook, but the code will run if you are programming a simple script from your text editor. You'll just need to adapt it (it's not hard).

The requirements that we'll need to install are:
* [NumPy](http://www.numpy.org/): This is the fundamental package for scientific computing with Python. Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data.
* [Pandas](http://pandas.pydata.org/): This is an open source library providing high-performance, easy-to-use data structures and data analysis tools.
* [Tweepy](http://www.tweepy.org/): This is an easy-to-use Python library for accessing the Twitter API.
* [Matplotlib](http://matplotlib.org/): This is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* [Seaborn](https://seaborn.pydata.org/): This is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
* [Textblob](https://textblob.readthedocs.io/): This is a Python library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks.

All of them are "*pip installable*". At the end of this article you'll be able to find more references about this Python libraries.

**Now that we have all the requirements, let's get started!**

# 1. Extracting twitter data (tweepy + pandas)

### 1.1. Importing our libraries

This will be the most difficult part of all the post... 😥
Just kidding, obviously it won't. It'll be just as easy as copying and pasting the following code in your notebook:

```python
# General:
import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

Excellent! We can now just run this cell of code and go to the next subsection.

### 1.2. Creating a Twitter App

In order to extract tweets for a posterior analysis, we need to access to our Twitter account and create an app. The website to do this is [https://apps.twitter.com/](https://apps.twitter.com/). (If you don't know how to do this, you can follow this [tutorial video](https://www.youtube.com/watch?v=BOA7SD_09Qk) to create an account and an application.)

From this app that we're creating we will save the following information in a script called `credentials.py`:
* Consumer Key (API Key)
* Consumer Secret (API Secret)
* Access Token
* Access Token Secret

An example of this script is the following:
```python
# Twitter App access keys for @user

# Consume:
CONSUMER_KEY    = ''
CONSUMER_SECRET = ''

# Access:
ACCESS_TOKEN  = ''
ACCESS_SECRET = ''
```

The reason of creating this extra file is that we want to export only the value of this variables, but being unseen in our main code (our notebook). We are now able to consume Twitter's API. In order to do this, we will create a function to allow us our keys authentication. We will add this function in another cell of code and we will run it:

```python
# We import our access keys:
from credentials import *    # This will allow us to use the keys as variables

# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api
```

So far, so easy right? We're good to extract tweets in the next section.


### 1.3. Tweets extraction

Now that we've created a function to setup the Twitter API, we can use this function to create an "*extractor*" object. After this, we will use Tweepy's function `extractor.user_timeline(screen_name, count)` to extract from `screen_name`'s user the quantity of `count` tweets.

As it is mentioned in the title, I've chosen [@realDonaldTrump](https://twitter.com/realDonaldTrump) as the user to extract data for a posterior analysis. Yeah, we wanna keep it interesting, LOL.

The way to extract Twitter's data is as follows:

```python
# We create an extractor object:
extractor = twitter_setup()

# We create a tweet list as follows:
tweets = extractor.user_timeline(screen_name="realDonaldTrump", count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()
```

With this we will have an output similar to this one, and we are able to compare the output with the Twitter account (to check if we're being consistent):
<code>Number of tweets extracted: 200.</code>
<code>5 recent tweets:</code>
<code>On behalf of @FLOTUS Melania & myself, THANK YOU for today's update & GREAT WORK! #SouthernBaptist @SendRelief,… https://t.co/4yZCeXCt6n</code>
<code>I will be going to Texas and Louisiana tomorrow with First Lady. Great progress being made! Spending weekend working at White House.</code>
<code>Stock Market up 5 months in a row!</code>
<code>'President Donald J. Trump Proclaims September 3, 2017, as a National Day of Prayer' #HurricaneHarvey #PrayForTexas… https://t.co/tOMfFWwEsN</code>
<code>Texas is healing fast thanks to all of the great men & women who have been working so hard. But still so much to do. Will be back tomorrow!</code>

We now have an extractor and extracted data, which is listed in the `tweets` variable. I must mention at this point that each element in that list is a `tweet` object from Tweepy, and we will learn how to handle this data in the next subsection.

### 1.4. Creating a (pandas) DataFrame

We now have initial information to construct a pandas `DataFrame`, in order to manipulate the info in a very easy way.

IPython's `display` function plots an output in a friendly way, and the `head`method of a dataframe allows us to visualize the first 5 elements of the dataframe (or the first number of elements that are passed as an argument).

So, using Python's list comprehension:

```python
# We create a pandas dataframe as follows:
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

# We display the first 10 elements of the dataframe:
display(data.head(10))
```

This will create an output similar to this:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>On behalf of @FLOTUS Melania & myself, THA...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I will be going to Texas and Louisiana tomorro...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stock Market up 5 months in a row!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'President Donald J. Trump Proclaims September...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas is healing fast thanks to all of the gre...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>...get things done at a record clip. Many big ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>General John Kelly is doing a great job as Chi...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wow, looks like James Comey exonerated Hillary...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>THANK YOU to all of the incredible HEROES in T...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RT @FoxNews: .@KellyannePolls on Harvey recove...</td>
    </tr>
  </tbody>
</table>

So we now have a nice table with ordered data.

An interesting thing is the number if internal methods that the `tweet`structure has in Tweepy:

```python
# Internal methods of a single tweet object:
print(dir(tweets[0]))
```

This outputs the following list of elements:
<code>['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_api', '_json', 'author', 'contributors', 'coordinates', 'created_at', 'destroy', 'entities', 'favorite', 'favorite_count', 'favorited', 'geo', 'id', 'id_str', 'in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'is_quote_status', 'lang', 'parse', 'parse_list', 'place', 'possibly_sensitive', 'retweet', 'retweet_count', 'retweeted', 'retweets', 'source', 'source_url', 'text', 'truncated', 'user']</code>

The interesting part from here is the quantity of metadata contained in a single tweet. If we want to obtain data such as the creation date, or the source of creation, we can access the info with this attributes. An example is the following:

```python
# We print info from the first tweet:
print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)
```

Obtaining an output like this:
<code>903778130850131970</code>
<code>2017-09-02 00:34:32</code>
<code>Twitter for iPhone</code>
<code>24572</code>
<code>5585</code>
<code>None</code>
<code>None</code>
<code>{'hashtags': [{'text': 'SouthernBaptist', 'indices': [90, 106]}], 'symbols': [], 'user_mentions': [{'screen_name': 'FLOTUS', 'name': 'Melania Trump', 'id': 818876014390603776, 'id_str': '818876014390603776', 'indices': [13, 20]}, {'screen_name': 'sendrelief', 'name': 'Send Relief', 'id': 3228928584, 'id_str': '3228928584', 'indices': [107, 118]}], 'urls': [{'url': 'https://t.co/4yZCeXCt6n', 'expanded_url': 'https://twitter.com/i/web/status/903778130850131970', 'display_url': 'twitter.com/i/web/status/9…', 'indices': [121, 144]}]}</code>

We're now able to order the relevant data and add it to our dataframe.


### 1.5. Adding relevant info to our dataframe

As we can see, we can obtain a lot of data from a single tweet. But not all this data is always useful for specific stuff. In our case we well just add some data to our dataframe. For this we will use Pythons list comprehension and a new column will be added to the dataframe by just simply adding the name of the content between square brackets and assign the content. The code goes as...:

```python
# We add relevant data:
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])
```

And to display again the dataframe to see the changes we just...:
```python
# Display of first 10 elements from dataframe:
display(data.head(10))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
      <th>len</th>
      <th>ID</th>
      <th>Date</th>
      <th>Source</th>
      <th>Likes</th>
      <th>RTs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>On behalf of @FLOTUS Melania & myself, THA...</td>
      <td>144</td>
      <td>903778130850131970</td>
      <td>2017-09-02 00:34:32</td>
      <td>Twitter for iPhone</td>
      <td>24572</td>
      <td>5585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I will be going to Texas and Louisiana tomorro...</td>
      <td>132</td>
      <td>903770196388831233</td>
      <td>2017-09-02 00:03:00</td>
      <td>Twitter for iPhone</td>
      <td>44748</td>
      <td>8825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stock Market up 5 months in a row!</td>
      <td>34</td>
      <td>903766326631698432</td>
      <td>2017-09-01 23:47:38</td>
      <td>Twitter for iPhone</td>
      <td>44518</td>
      <td>9134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'President Donald J. Trump Proclaims September...</td>
      <td>140</td>
      <td>903705867891204096</td>
      <td>2017-09-01 19:47:23</td>
      <td>Media Studio</td>
      <td>47009</td>
      <td>15127</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas is healing fast thanks to all of the gre...</td>
      <td>143</td>
      <td>903603043714957312</td>
      <td>2017-09-01 12:58:48</td>
      <td>Twitter for iPhone</td>
      <td>77680</td>
      <td>15398</td>
    </tr>
    <tr>
      <th>5</th>
      <td>...get things done at a record clip. Many big ...</td>
      <td>113</td>
      <td>903600265420578819</td>
      <td>2017-09-01 12:47:46</td>
      <td>Twitter for iPhone</td>
      <td>54664</td>
      <td>11424</td>
    </tr>
    <tr>
      <th>6</th>
      <td>General John Kelly is doing a great job as Chi...</td>
      <td>140</td>
      <td>903597166249246720</td>
      <td>2017-09-01 12:35:27</td>
      <td>Twitter for iPhone</td>
      <td>59840</td>
      <td>11678</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wow, looks like James Comey exonerated Hillary...</td>
      <td>130</td>
      <td>903587428488839170</td>
      <td>2017-09-01 11:56:45</td>
      <td>Twitter for iPhone</td>
      <td>110667</td>
      <td>35936</td>
    </tr>
    <tr>
      <th>8</th>
      <td>THANK YOU to all of the incredible HEROES in T...</td>
      <td>110</td>
      <td>903348312421670912</td>
      <td>2017-08-31 20:06:35</td>
      <td>Twitter for iPhone</td>
      <td>112012</td>
      <td>29064</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RT @FoxNews: .@KellyannePolls on Harvey recove...</td>
      <td>140</td>
      <td>903234878124249090</td>
      <td>2017-08-31 12:35:50</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>6638</td>
    </tr>
  </tbody>
</table>

Now that we have extracted and have the data in a easy-to-handle ordered way, we're ready to do a bit more of manipulation to visualize some plots and gather some statistical data. The first part of the post is done.

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

We're almost done with this second section of the post. Now we will plot the sources in a pie chart, since we realized that not every tweet is tweeted from the same source (😱🤔). We first clean all the sources:

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

# 3. Sentiment analysis

### 3.1. Importing textblob

As we mentioned at the beginning of this post, textblob will allow us to do sentiment analysis in a very simple way. We will also use the `re` library from Python, which is used to work with *regular expressions*. For this, I'll provide you two utility functions to: a) clean text (which means that any symbol distinct to an alphanumeric value will be remapped into a new one that satisfies this condition), and b) create a classifier to analyze the polarity of each tweet after cleaning the text in it. I won't explain the specific way in which the function that cleans works, since it would be extended and it might be better understood in the [official `re`documentation](https://docs.python.org/3/library/re.html).

The code that I'm providing is:
```python
from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
```

The way it works is that textblob already provides a trained analyzer (cool, right?). Textblob can work with different *machine learning* models used in *natural language processing*. If you want to train your own classifier (or at least check how it works) feel free to check the following [link](https://textblob.readthedocs.io/en/dev/classifiers.html). It might result relevant since we're working with a pre-trained model (for which we don't not the data that was used).

Anyway, getting back to the code we will just add an extra column to our data. This column will contain the sentiment analysis and we can plot the dataframe to see the update:

```python
# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# We display the updated dataframe with the new column:
display(data.head(10))
```

Obtaining the new output:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
      <th>len</th>
      <th>ID</th>
      <th>Date</th>
      <th>Source</th>
      <th>Likes</th>
      <th>RTs</th>
      <th>SA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>On behalf of @FLOTUS Melania & myself, THA...</td>
      <td>144</td>
      <td>903778130850131970</td>
      <td>2017-09-02 00:34:32</td>
      <td>Twitter for iPhone</td>
      <td>24572</td>
      <td>5585</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I will be going to Texas and Louisiana tomorro...</td>
      <td>132</td>
      <td>903770196388831233</td>
      <td>2017-09-02 00:03:00</td>
      <td>Twitter for iPhone</td>
      <td>44748</td>
      <td>8825</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Stock Market up 5 months in a row!</td>
      <td>34</td>
      <td>903766326631698432</td>
      <td>2017-09-01 23:47:38</td>
      <td>Twitter for iPhone</td>
      <td>44518</td>
      <td>9134</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>'President Donald J. Trump Proclaims September...</td>
      <td>140</td>
      <td>903705867891204096</td>
      <td>2017-09-01 19:47:23</td>
      <td>Media Studio</td>
      <td>47009</td>
      <td>15127</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas is healing fast thanks to all of the gre...</td>
      <td>143</td>
      <td>903603043714957312</td>
      <td>2017-09-01 12:58:48</td>
      <td>Twitter for iPhone</td>
      <td>77680</td>
      <td>15398</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>...get things done at a record clip. Many big ...</td>
      <td>113</td>
      <td>903600265420578819</td>
      <td>2017-09-01 12:47:46</td>
      <td>Twitter for iPhone</td>
      <td>54664</td>
      <td>11424</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>General John Kelly is doing a great job as Chi...</td>
      <td>140</td>
      <td>903597166249246720</td>
      <td>2017-09-01 12:35:27</td>
      <td>Twitter for iPhone</td>
      <td>59840</td>
      <td>11678</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wow, looks like James Comey exonerated Hillary...</td>
      <td>130</td>
      <td>903587428488839170</td>
      <td>2017-09-01 11:56:45</td>
      <td>Twitter for iPhone</td>
      <td>110667</td>
      <td>35936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>THANK YOU to all of the incredible HEROES in T...</td>
      <td>110</td>
      <td>903348312421670912</td>
      <td>2017-08-31 20:06:35</td>
      <td>Twitter for iPhone</td>
      <td>112012</td>
      <td>29064</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RT @FoxNews: .@KellyannePolls on Harvey recove...</td>
      <td>140</td>
      <td>903234878124249090</td>
      <td>2017-08-31 12:35:50</td>
      <td>Twitter for iPhone</td>
      <td>0</td>
      <td>6638</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

As we can see, the last column contains the sentiment analysis (`SA`). We now just need to check the results.

### 3.2. Analyzing the results

To have a simple way to verify the results, we will count the number of neutral, positive and negative tweets and extract the percentages.

```python
# We construct lists with classified tweets:

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]
```

Now that we have the lists, we just print the percentages:
```print
# We print percentages:

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
```

Obtaining the following result:
`Percentage of positive tweets: 51.0%`
`Percentage of neutral tweets: 27.0%`
`Percentage de negative tweets: 22.0%`

We have to consider that we're working only with the 200 most recent tweets from D. Trump (last updated: September 2nd.). For more accurate results we can consider more tweets. An interesting thing (an invitation to the readers) is to analyze the polarity of the tweets from different sources, it might be deterministic that by only considering the tweets from one source the polarity would result more positive/negative. Anyway, I hope this resulted interesting.

As we saw, we can extract, manipulate, visualize and analyze data in a very simple way with Python. I hope that this leaves some uncertainty in the reader, for further exploration using this tools.

It might be possible to find little mistakes in the translation of the material (I designed the workshop in Spanish, originally 😅). Please feel free to comment or suggest all that comes up to your mind. That would complement some ideas that I already have in mind for further work. 😀

I'll now leave some references for documentation and tutorials on the used libraries. Hope to hear from you!

## References:
<div class="text_cell_render border-box-sizing rendered_html">
<ol>
<li>
<a href="http://tweepy.readthedocs.io/en/v3.5.0/">Official documentation - Tweepy</a>.</li>
<li>
<a href="https://docs.scipy.org/doc/numpy-dev/index.html">Official documentation - NumPy</a>.</li>
<li>
<a href="https://docs.scipy.org/doc/numpy-dev/user/quickstart.html">Official tutorial - NumPy</a>.</li>
<li>
<a href="https://pandas.pydata.org/pandas-docs/stable/tutorials.html">Official tutorial - Pandas</a>.</li>
<li>
<a href="https://pandas.pydata.org/pandas-docs/stable/index.html">Official documentation - Pandas</a>.</li>
<li>
<a href="http://matplotlib.org/index.html">Official documentation - Matplotlib</a>.</li>
<li>
<a href="http://matplotlib.org/users/pyplot_tutorial.html">Official tutorial - Pyplot</a>.</li>
<li>
<a href="https://seaborn.pydata.org/">Official website - Seaborn</a>.</li>
<li>
<a href="https://textblob.readthedocs.io/en/dev/">Official documentation - TextBlob</a>.</li>
<li>
<a href="https://textblob.readthedocs.io/en/dev/classifiers.html">Tutorial: Building a Text Classification System - TextBlob</a>.</li>
</ol>
</div>
