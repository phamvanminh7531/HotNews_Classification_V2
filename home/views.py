from django.http import response
from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from home.re_processing import train_re_processing
from home.re_processing import redict_processing
from sklearn.naive_bayes import MultinomialNB
import requests
from bs4 import BeautifulSoup

# Đọc file csv
df = pd.read_csv('home/data/hot_news_data.csv', encoding='utf-8-sig')
raw_content = df['content']
X = train_re_processing(raw_content=raw_content)
y = df['label']

# Chuyển dữ liệu các từ thành ma trận
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X)
freq_term_matrix = count_vectorizer.transform(X)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

# Tạo các tập train và test để tiến hành huấn luyện
x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix,y, random_state=0, test_size=0.4)

# Huấn luyện model NAIVE BAYES để phân loại các bài báo
model = MultinomialNB(alpha=0)
model.fit(x_train, y_train)
Accuracy = model.score(x_test, y_test)

class HotNews():
    def __init__(self, title, content, image, date_time, author,link, label):
        self.title = title
        self.content = content
        self.image = image
        self.date_time = date_time
        self.author = author
        self.link = link
        self.label = label

# Create your views here.
def home(request):
    # crawl data
    list_hot_news = []
    list_news = []
    response = requests.get("https://tuoitre.vn/tin-moi-nhat.htm")
    soup = BeautifulSoup(response.content, "html.parser")
    titles = soup.findAll('h3', class_='title-news')
    links = [link.find('a').attrs["href"] for link in titles]
    for link in links:
        link_a = "https://tuoitre.vn" + link
        news = requests.get("https://tuoitre.vn" + link)
        soup = BeautifulSoup(news.content, "html.parser")
        title = soup.find("h1", class_="article-title").text
        content = soup.find("h2", class_="sapo").text
        try:
            body = soup.find("div", id="main-detail-body")
            image = body.find("img").attrs["src"]
        except:
            image = ''
        date_time = soup.find("div", class_="date-time").text
        author = soup.find("div", class_="author").text
        label = predict(title + " " +content)
        new_news = HotNews(title,content,image,date_time,author,link_a,label)
        
        # nếu dự đoán ra 1 (hot news) thì cho vào list hot news
        if label==1:
            list_hot_news.append(new_news)
        # nếu dự đoán ra 0 (hot news) thì cho vào list news
        else:
            list_news.append(new_news)
    context = {
        "hot_news":list_hot_news,
        "news":list_news
    }
    return render(request, "home/index.html",context=context)

def predict(raw_news):
    news = redict_processing(raw_news)
    news_vector = count_vectorizer.transform(news)
    news_matrix = tfidf.fit_transform(news_vector)
    result = model.predict(news_matrix)[0]
    return result