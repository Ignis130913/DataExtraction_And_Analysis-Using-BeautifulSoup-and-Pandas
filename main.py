import pandas as pd, requests
from bs4 import BeautifulSoup
# from textblob import TextBlob
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('cmudict')

import re

sid=SentimentIntensityAnalyzer()

def extract_content(url):
    try:
        response=requests.get(url)
        soup=BeautifulSoup(response.text,'html.parser')
        heading=soup.find('h1').text.strip() if soup.find('h1') else ''
        text=''.join([p.text.strip() for p in soup.find_all('p')])
        return heading,text
    except Exception as e:
        return None,None

not_exist=[]

def extract(file):
    articles=[]
    df=pd.read_excel(file)
    for index,row in df.iterrows():
        url_id=row['URL_ID']
        url=row['URL']
        heading,text=extract_content(url)
        if heading and text:
            text=''.join(text.split('.')[:-4])
            articles.append({'url_id':url_id,'heading':heading,'text':text})
        else: not_exist.append(url_id)
    return df,articles

print("Starting Process")
file='Input.xlsx'
df,articles=extract(file)
print("Extraction Done")
for u in not_exist: df=df.drop(df[df['URL_ID'] == u].index)

# print(len(articles))
# for article in articles:
#     print(article['heading'])
# print(df)
output={
    'positive_score':[],
    'negative_score':[],
    'polarity_score':[],
    'subjectivity_score':[],
    'avg_sentence_length':[],
    'percentage_complex_words':[],
    'fog_index':[],
    'avg_wps':[],
    'complex_wcnt':[],
    'wcnt':[],
    'syllable_pword':[],
    'personal_pronouns':[],
    'avg_word_len':[]
}

def remove_stopwords(text):
    stop_words=set(stopwords.words('english'))
    paragraphs=text.split('.')
    filtered_paragraphs=[]
    for paragraph in paragraphs:
        word_tokens=word_tokenize(paragraph)
        filtered_words=[word for word in word_tokens if word.lower() not in stop_words]
        filtered_paragraph=' '.join(filtered_words).strip()
        if filtered_paragraph:
            filtered_paragraphs.append(filtered_paragraph)
    return '. '.join(filtered_paragraphs)

# def get_positive_score(text):
#     blob=TextBlob(text)
#     positive_score=0
#     for sentence in blob.sentences:
#         if sentence.sentiment.polarity>0: positive_score+=1
#     return positive_score

# def get_negative_score(text):
#     blob=TextBlob(text)
#     negative_score=0
#     for sentence in blob.sentences:
#         #print(sentence.sentiment.polarity)
#         if sentence.sentiment.polarity<0: negative_score+=1
#     return negative_score

# def get_polarity(text):
#     return TextBlob(text).sentiment.polarity

# def get_subjectivity_score(text):
#     return TextBlob(text).sentiment.subjectivity

def get_posNegPolSub(text):
    scores=sid.polarity_scores(text)
    return scores['pos'],scores['neg'],scores['neu'],scores['compound']

def get_avg_sentence_length(text):
    sentences=re.split(r'[.!?]',text)
    sentences=[sentence.strip() for sentence in sentences]
    total_words=sum(len(sentence.split()) for sentence in sentences)
    return total_words/len(sentences)
    
# def get_percentage_of_complex_words(text):
#     words=text.split()
#     def count_syllables(word):
#             cmu_dict = cmudict.dict()
#             syllables = cmu_dict.get(word.lower())
#             if syllables:
#                 return max(len(list(y for y in x if y[-1].isdigit())) for x in syllables)
#             return 0
#     complex_word_cnt=sum(1 for word in words if count_syllables(word)>=3)
#     return complex_word_cnt,(complex_word_cnt/len(words))*100

def get_percentage_of_complex_words(text):
    words=text.split()
    cmu_dict=cmudict.dict()
    complex_word_cnt=0

    def count_syllables(word):
        syllables=cmu_dict.get(word.lower())
        if syllables:
            # Simplified syllable counting based on vowel sequences
            return sum(1 for phoneme in syllables[0] if re.match(r'[aeiouAEIOU]', phoneme))
        return 0

    for word in words:
        if count_syllables(word)>=3:
            complex_word_cnt+=1

    percentage_complex_words=(complex_word_cnt / len(words)) * 100
    return complex_word_cnt, percentage_complex_words

def get_avg_wps(text):
    sentences=re.split(r'[.!?]',text)
    sentences=[sentence.strip() for sentence in sentences]
    total_words=sum(len(sentence.split()) for sentence in sentences)
    return total_words/len(sentences)

def get_word_count(text):
    return len(text.split())

# def get_average_syllable_per_word(text):
#     words=text.split()
#     def syllable_cnt(word):
#         cmu_dict=cmudict.dict()
#         if word.lower() in cmu_dict: return max([len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]])
#         return 0
#     total_syllables=sum(syllable_cnt(word) for word in words)
#     return total_syllables/len(words)

def get_average_syllable_per_word(text):
    words=text.split()
    cmu_dict=cmudict.dict()
    total_syllables=0

    def syllable_count(word):
        syllables=cmu_dict.get(word.lower())
        if syllables:
            # Simplified syllable counting based on vowel sequences
            return sum(1 for phoneme in syllables[0] if phoneme[-1].isdigit())
        return 0

    for word in words:
        total_syllables+=syllable_count(word)

    average_syllables_per_word=total_syllables / len(words)
    return average_syllables_per_word

def get_personal_pronouns(text):
    text=text.lower()
    personal_pronouns = ['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourselves', 'they', 'them', 'their', 'theirs', 'themselves']
    return sum(text.count(pronoun) for pronoun in personal_pronouns)

def get_avg_word_length(text):
    words=text.split()
    total_len=sum(len(word) for word in words)
    return total_len/len(words)

cnt=1

print("Starting Calculation")
for article in articles:
    text=remove_stopwords(article['text'].strip())
    # positive_score=get_positive_score(text)
    # nagative_score=get_negative_score(text)
    # polarity_score=get_polarity(text)
    # subjectivity_score=get_subjectivity_score(text)
    positive_score,negative_score,subjectivity_score,polarity_score=get_posNegPolSub(text)
    avg_sentence_len=get_avg_sentence_length(text)
    complex_word_cnt,percentage_of_complex_words=get_percentage_of_complex_words(text)
    fog_index=0.4*(avg_sentence_len+percentage_of_complex_words)
    avg_wps=get_avg_wps(text)
    word_cnt=get_word_count(text)
    avg_syllable=get_average_syllable_per_word(text)
    personal_pronouns=get_personal_pronouns(text)
    avg_word_len=get_avg_word_length(text)
    #print(positive_score,nagative_score,polarity_score,subjectivity_score,avg_sentence_len,percentage_of_complex_words,complex_word_cnt, fog_index,avg_wps,word_cnt,avg_syllable,personal_pronouns,avg_word_len)
    output['positive_score'].append(positive_score)
    output['negative_score'].append(negative_score)
    output['polarity_score'].append(polarity_score)
    output['subjectivity_score'].append(subjectivity_score)
    output['avg_sentence_length'].append(avg_sentence_len)
    output['percentage_complex_words'].append(percentage_of_complex_words)
    output['fog_index'].append(fog_index)
    output['avg_wps'].append(avg_wps)
    output['complex_wcnt'].append(complex_word_cnt)
    output['wcnt'].append(word_cnt)
    output['syllable_pword'].append(avg_syllable)
    output['personal_pronouns'].append(personal_pronouns)
    output['avg_word_len'].append(avg_word_len)
    
    print(f"{cnt} Row 's Completed")
    cnt+=1

print("Calculation Done")
    
df['POSITIVE SCORE']=output['positive_score']
df['NEGATIVE SCORE']=output['negative_score']
df['POLARITY SCORE']=output['polarity_score']
df['SUBJECTIVITY SCORE']=output['subjectivity_score']
df['AVG SENTENCE LENGTH']=output['avg_sentence_length']
df['PERCENTAGE OF COMPLEX WORDS']=output['percentage_complex_words']
df['FOG INDEX']=output['fog_index']
df['AVG NUMBER OF WORDS PER SENTENCE']=output['avg_wps']
df['COMPLEX WORD COUNT']=output['complex_wcnt']
df['WORD COUNT']=output['wcnt']
df['SYLLABLE PER WORD']=output['syllable_pword']
df['PERSONAL PRONOUNS']=output['personal_pronouns']
df['AVG WORD LENGTH']=output['avg_word_len']

#print(df)
print("Creating Output File")
op_file_path='Output.xlsx'
df.to_excel(op_file_path,index=False)

# print(df.shape)

# for k,v in output.items():
#     print(k,v)

