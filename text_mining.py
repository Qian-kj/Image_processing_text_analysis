import pandas as pd
import re
import matplotlib.pyplot as plt
import time

# start time
start_time =time.time()

DEBUGGING = False
TOP_MOST = 10

# data path
DATA_DIR  = 'E:/KCL/Data mining/Coursework_2/Data_mining_cousework_2_AB34774/data/text_data/'
DATA_FILE = 'Corona_NLP_train.csv'
STOP_WORDS = 'stopwords.txt'

# read in the dataframe
raw_data = pd.read_csv(DATA_DIR + DATA_FILE, encoding='latin1')

# 1.1
# the possible sentiments
stmt_count = raw_data.groupby(["Sentiment"], as_index = False).size()
sort_stmt = stmt_count.sort_values(by="size", ascending=False)
for i in range(len(sort_stmt)):
    sentiment = sort_stmt.iat[i,0]
    count_sentimet = sort_stmt.iat[i,1]
    print("There is a sentiment: {}, count = {}".format(sentiment, count_sentimet))
# the second most popular sentiment in the tweets
sec_sentiment = sort_stmt.iat[1,0]
count_sec_sentiment = sort_stmt.iat[1,1]
print("The second most popular sentiment is {}, and its count is {}.".format(sec_sentiment,count_sec_sentiment))

# the date that has the greatest number of extremely positive tweets
pos_data = raw_data[raw_data["Sentiment"] == "Extremely Positive"]
pos_count = pos_data.groupby(["TweetAt"], as_index=False).size()
sort_pos = pos_count.sort_values(by="size", ascending=False)
EP_date = sort_pos.iat[0,0]
EP_count = sort_pos.iat[0,1]
print("In {}, the largest number ({}) of extremely positive tweets had been posted.".format(EP_date, EP_count))

# convert the messages to lower case
raw_data["OriginalTweet"] = raw_data["OriginalTweet"].str.lower()

# replace non-alphabetical characters with whitespaces
raw_data["OriginalTweet"] = [re.sub("[^A-Za-z]", " " , i) for i in raw_data["OriginalTweet"]]
texs = raw_data["OriginalTweet"]

# ensure that the words of a message are separated by a single whitespace.
raw_data["OriginalTweet"] = [re.sub(" +", " ", line) for line in texs]
texts = raw_data["OriginalTweet"]

# 1.2
# tokenize the tweets
all_word = []
wordList = [text.strip() for text in texts]
for line in wordList:
    word = line.split(" ")
    all_word.extend(word)

# count the total number of all words (including repetitions)
print("The total number of all words (including repetitions):", len(all_word))

# count the number of all distinct words
print("the number of all distinct words", len(set(all_word)))

# count the 10 most frequent words
all_words_df = pd.DataFrame(all_word, columns=["words"])
words_count = all_words_df.groupby(["words"], as_index=False).size()
sort_words = words_count.sort_values(by="size", ascending=False)
for i in range(10):
    ten_words = sort_words.iat[i,0]
    count_ten_words = sort_words.iat[i,1]
    print("The top {} frequency words: {}, count = {}".format((i + 1), ten_words, count_ten_words))

# remove stop words, words with 2 characters and recalculate the number of all words
# read stop words
stopwords = []
with open(DATA_DIR + STOP_WORDS) as f:
    for line in f.readlines():
        word = line.strip()
        stopwords.append(word)
print('number of stopwords = ' + str(len(stopwords)))
if (DEBUGGING):
    print('stopwords=', stopwords)

# drop stopwords from the dataframe
words_clean_df = all_words_df.copy()
stopwords_df = words_clean_df[words_clean_df.words.isin(stopwords)]
words_clean_df = words_clean_df.drop(stopwords_df.index)

# select words who contains two more characters
def clean_def(x):
    if (len(str(x)) > 2):
        clean_word = str(x)
        return clean_word
words_clean_df['words'] = words_clean_df['words'].apply(clean_def)

# change the column name to clean words
words_clean_df = words_clean_df.rename(columns = {"words" : "clean words"})

# count the number of all words
print("the number of all words (including repetitions) after processing:", len(words_clean_df))

# recalculate all words (including repetitions) and the 10 most frequent words in the modified corpus.
words_clean_count = words_clean_df.groupby(["clean words"], as_index = False).size()
sort_words_clean = words_clean_count.sort_values(by="size", ascending=False)
for i in range(10):
    ten_words_clean = sort_words_clean.iat[i,0]
    count = sort_words_clean.iat[i,1]
    print("The top {} frequency words: {}, count = {}".format((i + 1), ten_words_clean, count))

# 1.3
fig = plt.figure(figsize=(12,6))
# plot the top 1-14 frequency words
sort_words_plot = sort_words_clean.iloc[0:14].sort_values(by="size", ascending=True)
plt.plot(sort_words_plot.iloc[0:14,0], sort_words_plot.iloc[0:14,1]/len(words_clean_df))
plt.tick_params(axis='x', labelsize=8)
plt.xticks(rotation=45)
plt.title("The top 1-14 frequency words")
plt.xlabel("Words")
plt.ylabel("The count of words")
plt.grid()
plt.savefig(DATA_DIR + 'The top 1-14 frequency words.jpg')
plt.show()


# 1.4
from sklearn.feature_extraction.text import CountVectorizer as CVT
from sklearn.naive_bayes import MultinomialNB

# tweets data set
X_data = raw_data["OriginalTweet"].copy()
# sentiment data set
y = raw_data["Sentiment"].copy()

# using term frequency–inverse document frequency for the data set using countvectorizer
cvt = CVT().fit(X_data)
X = cvt.transform(X_data)

# build an Multinomial Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, y)
main_accucy = clf.score(X,y)
print("Error rate", (1 - main_accucy))

# count time
end_time = time.time()
print('Running time: {}s Seconds'.format(end_time-start_time))