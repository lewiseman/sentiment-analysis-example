import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

comments = [
    'I love you',
    'I hate you',
    'I dont know'
]

x_axis = list(range(1,4))
negative_axis = []
neutral_axis = []
positive_axis = []

for comment in comments:
    ss = sid.polarity_scores(comment)
    for a,b in ss.items():
        if a == 'neg':
            negative_axis.append(ss[a])
        elif a == 'neu':
            neutral_axis.append(ss[a])
        elif a == 'pos':
            positive_axis.append(ss[a])


plt.plot(x_axis, negative_axis, color='red')
plt.plot(x_axis, positive_axis, color='green')
plt.plot(x_axis, neutral_axis, color='black')
plt.title('Davis and Shirltliff Facebook Sentiment Analysis')
plt.xlabel('Time(s)')
plt.ylabel('sentiment score')
plt.tight_layout()
plt.show()