import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')

def extract_content(url):
    try:
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        heading=soup.find('h1').text.strip()if soup.find('h1')else''
        paragraphs=[p.text.strip()for p in soup.find_all('p')]
        text=' '.join(paragraphs)
        return heading,text
    except Exception as e:
        return None,None

not_exist=[]

def extract(file):
    articles=[]
    df=pd.read_excel(file)
    for index,row in df.iterrows():
        url_id,url=row['URL_ID'],row['URL']
        heading,text=extract_content(url)
        if heading and text:
            text='.'.join(text.split('.')[:-4])
            articles.append({'url_id':url_id,'heading':heading,'text':text})
        else:
            # print(f"Content not found for URL_ID: {url_id}")
            not_exist.append(url_id)
    return df,articles

def remove_stopwords(text):
    stop_words=set(stopwords.words('english'))
    word_tokens=word_tokenize(text)
    filtered_text=[word for word in word_tokens if word.lower()not in stop_words]
    return' '.join(filtered_text)

def analyze_text(text):
    blob=TextBlob(text)
    positive_score=sum(1 for sentence in blob.sentences if sentence.sentiment.polarity>0)
    negative_score=sum(1 for sentence in blob.sentences if sentence.sentiment.polarity<0)
    polarity_score=blob.sentiment.polarity
    subjectivity_score=blob.sentiment.subjectivity
    avg_sentence_length=len(re.findall(r'\w+',text))/len(re.findall(r'[.!?]',text))
    word_tokens=word_tokenize(text)
    complex_word_cnt=sum(1 for word in word_tokens if len(word)>6)
    word_count=len(word_tokens)
    avg_word_length=sum(len(word)for word in word_tokens)/word_count
    return positive_score,negative_score,polarity_score,subjectivity_score,avg_sentence_length,complex_word_cnt,word_count,avg_word_length

def process_article(article):
    text=article['text'].strip()
    text=remove_stopwords(text)
    positive_score,negative_score,polarity_score,subjectivity_score,avg_sentence_length,complex_word_cnt,word_count,avg_word_length=analyze_text(text)
    return{'url_id':article['url_id'],'heading':article['heading'],
           'positive_score':positive_score,'negative_score':negative_score,
           'polarity_score':polarity_score,'subjectivity_score':subjectivity_score,
           'avg_sentence_length':avg_sentence_length,'complex_word_count':complex_word_cnt,
           'word_count':word_count,'avg_word_length':avg_word_length}

def main():
    input_file='Input.xlsx'
    output_file='Output.xlsx'

    df,articles=extract(input_file)
    for u in not_exist: df=df.drop(df[df['URL_ID'] == u].index)
    processed_articles=[process_article(article)for article in articles]

    processed_df=pd.DataFrame(processed_articles)
    processed_df=df.merge(processed_df,on='url_id')

    processed_df.to_excel(output_file,index=False)
    print("Processing completed.")

if __name__=="__main__":
    main()
