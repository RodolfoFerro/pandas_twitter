# 3. Sentiment analysis

### 3.1. Importing textblob

As we mentioned at the beginning of this workshop, textblob will allow us to do sentiment analysis in a very simple way. We will also use the `re` library from Python, which is used to work with *regular expressions*. For this, I'll provide you two utility functions to: a) clean text (which means that any symbol distinct to an alphanumeric value will be remapped into a new one that satisfies this condition), and b) create a classifier to analyze the polarity of each tweet after cleaning the text in it. I won't explain the specific way in which the function that cleans works, since it would be extended and it might be better understood in the [official `re`documentation](https://docs.python.org/3/library/re.html).

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


> Go back to [2. Visualization and basic statistics](https://github.com/RodolfoFerro/pandas_twitter/blob/master/02-visualization.md)<br>
> Go next to [4. References](https://github.com/RodolfoFerro/pandas_twitter/blob/master/04-references.md)<br>
