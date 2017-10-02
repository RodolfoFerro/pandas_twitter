# 1. Extracting twitter data (tweepy + pandas)

### 1.1. Importing our libraries

This will be the most difficult part of all the workshop... 😥
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

Now that we have extracted and have the data in a easy-to-handle ordered way, we're ready to do a bit more of manipulation to visualize some plots and gather some statistical data. The first part of the workshop is done.

> Go back to [0. Prerequisite: What will we need?](https://github.com/RodolfoFerro/pandas_twitter/blob/master/00-prerequisite.md)<br>
> Go next to [2. Visualization and basic statistics](https://github.com/RodolfoFerro/pandas_twitter/blob/master/02-visualization.md)<br>
