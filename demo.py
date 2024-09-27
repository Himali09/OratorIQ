import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('QA/questions_with_important_terms.csv')
df_hr = pd.read_csv('QA/HRquestions.csv')

hr_questions = df_hr[df_hr['QuestionType']=='HR']

Web_dev = df[df['Question_Type'] == 'Web Development']
Java = df[df['Question_Type'] == 'Java Programming']
Python = df[df['Question_Type'] == 'Python Programming']
AIML = df[df['Question_Type'] == 'AIML']
Cyber = df[df['Question_Type'] == 'Cyber Security']
Dscience = df[df['Question_Type'] == 'Data Science']
Cpp = df[df['Question_Type'] == 'C++ Programming']
C = df[df['Question_Type'] == 'C Programming']

def question(choice):
    if choice == '1':
        #print("Web Development Interview: ")
        random_questions = Web_dev.sample(min(len(Web_dev), 5))
        #answers += record_answers(random_questions)
    elif choice == '2':
        #print("Java Programming Interview:")
        random_questions = Java.sample(min(len(Java), 5))
        #answers += record_answers(random_questions)
    elif choice == '3':
        #print("Python Programming Interview:")
        random_questions = Python.sample(min(len(Python), 5))
        #answers += record_answers(random_questions)
    elif choice == '4':
        #print("AIML Interview:")
            random_questions = AIML.sample(min(len(AIML), 5))
        #answers += record_answers(random_questions)
    elif choice == '5':
        #print("Cyber Security Interview:")
        random_questions = Cyber.sample(min(len(Cyber), 5))
        #answers += record_answers(random_questions)
    elif choice == '6':
            #print("Data Science Interview:")
            random_questions = Dscience.sample(min(len(Dscience), 5))
            #answers += record_answers(random_questions)
    elif choice == '7':
            #print("C++ Progr['Question'].tolist()amming Interview:")
            random_questions = Cpp.sample(min(len(Cpp), 5))
            #answers += record_answers(random_questions)
    elif choice == '8':
            #print("C Progr['Question'].tolist()amming Interview:")
            random_questions = C.sample(min(len(C), 5))
            #answers += record_answers(random_questions)
    return random_questions['Question'].tolist(),random_questions['Important_Terms'].tolist()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase conversion
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stopwords]  # Remove stop words and non-alphanumeric tokens
    return ' '.join(filtered_tokens)

def record_answer_and_extract_features(answer):
    # Feature extraction using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    preprocessed_answer = preprocess_text(answer)
            # Perform feature extraction
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_answer])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return feature_names

def calculate_similarity(dataset_answer, user_answer):
    vectorizer = CountVectorizer().fit_transform([user_answer, dataset_answer])

    # Calculate the cosine similarity between the two texts
    cosine_sim = cosine_similarity(vectorizer)

    # The cosine similarity value is in the first row and second column of the matrix
    similarity = cosine_sim[0][1]
    return similarity

def ask_question():
    random_questions = hr_questions.sample(min(len(hr_questions), 7))
    return random_questions['Questions'].tolist()

