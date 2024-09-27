from pydub import AudioSegment
import speech_recognition as sr
from gramformer import Gramformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

gf = Gramformer(models = 1, use_gpu=False) # 1=corrector, 2=detector

def get_text(audio):
    r = sr.Recognizer()    
    try:
        audio_data = AudioSegment.from_file(audio)
        audio_data.export(f"{audio}", format="wav")
        with sr.AudioFile(audio) as source:
            audio_data = r.record(source)
        recognized_text = r.recognize_google(audio_data)
        return recognized_text
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Unknown error occurred")

def get_speech_rate(text,audio):
    audio = AudioSegment.from_file(audio)
    # Calculate the duration of the audio in seconds
    audio_duration_seconds = len(audio)/1000 # Convert milliseconds to seconds
    print(audio_duration_seconds)
    # Count the number of words in the recognized text
    word_count = len(text.split())

    # Calculate the rate of speech in words per minute (WPM)
    speech_rate_wpm = (word_count / audio_duration_seconds) * 60

    if(speech_rate_wpm<150):
        status='Your speech rate is Slow.'
    elif(speech_rate_wpm>=150 and speech_rate_wpm <=160):
        status='Your speech rate is Optimal'
    else:
        status='Your speech rate is Fast'
    return (speech_rate_wpm,status)

def correct_spoken_grammar(text):

    sentence = text
    corrected_sentence = gf.correct(sentence, max_candidates=1)
    corrected_sentence = ' '.join(corrected_sentence)
    num_errors = len(gf.get_edits(sentence,corrected_sentence))
    score = 1.0 - (num_errors / len(sentence.split()))

    return (str(corrected_sentence),score)
    
def tone_analysis(text):

    # Create a TextBlob object
    blob = TextBlob(text)

    # Analyze sentiment (polarity ranges from -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive)
    sentiment_score = blob.sentiment.polarity

    # Define a function to interpret the sentiment score
    def interpret_sentiment(score):
        if score < 0:
            return "Negative"
        elif score == 0:
            return "Neutral"
        else:
            return "Positive"

    # Get the interpreted sentiment
    tone = interpret_sentiment(sentiment_score)

    return (tone,sentiment_score)

def detect_pauses(audio_file):
    sound = AudioSegment.from_file(audio_file)
    silence_threshold = -50  # Adjust this threshold according to your audio
    silence_duration = 500  # Minimum duration of silence in milliseconds
    long_pause_duration=3000

    chunks = []
    start = 0
    end = 0
    is_silence = False

    for i in range(len(sound)):
        if sound[i].dBFS < silence_threshold:
            if not is_silence:
                start = i
                is_silence = True
        elif is_silence:
            end = i
            if end - start >= silence_duration:
                chunks.append((start, end))
                is_silence = False
# Extract audio segments with detected long pauses
    long_pause_segments = []
    for start, end in chunks:
        if end - start >= long_pause_duration:
            long_pause_segments.append(sound[start:end])

    audio_duration_minutes = len(audio_file) / 60000.0  # Convert minutes
    avg=int(len(long_pause_segments)/audio_duration_minutes)
    if(avg>2):
        status='Too Many Pauses'
    else:
        status='Optimal speech'
    return (len(long_pause_segments),status)

def analyze_speech(text):
    # Tokenize the speech into sentences and words
    words = word_tokenize(text)

    # Calculate word count and sentence count
    word_count = len(words)

    # Remove stopwords and punctuation from the words
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Calculate word frequency distribution
    word_freq = FreqDist(words)
    return (word_count,word_freq)

def similarity(text,text_file):

    # Open the file for reading (by default, it opens in read-only mode)
    with open(text_file, "r") as file:
        # Read the entire content of the file into a string
        expected_text = file.read()
    # Create a CountVectorizer to convert text to a bag-of-words representation
    vectorizer = CountVectorizer().fit_transform([text, expected_text])

    # Calculate the cosine similarity between the two texts
    cosine_sim = cosine_similarity(vectorizer)

    # The cosine similarity value is in the first row and second column of the matrix
    similarity = cosine_sim[0][1]

    if(similarity<0.4):
        status='Speech is not the same.'
    elif(0.4<=similarity<0.75):
        status='Speech is almost the same.'
    else:
        status='Speech is accurate'
    return  (similarity,status)


    