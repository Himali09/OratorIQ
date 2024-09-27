# app.py
from flask import Flask, Blueprint,render_template,redirect, url_for,request, jsonify,session,Response
import audio_analysis as au
import random
import string
import cv2
from gaze_tracking import GazeTracking
import demo
import time
import numpy as np
from flask_pymongo import PyMongo
from pymongo import MongoClient
from datetime import datetime
import speech_recognition as sr
import uuid
import statistics
import logging
import json
from bson.objectid import ObjectId

# Set up logging
logging.basicConfig(level=logging.DEBUG)

#Loading the xml file for video capture
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gaze = GazeTracking()

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['User']

# Define users collection
users_collection=db['users']
reports_collection=db['reports']

app = Flask(__name__,static_folder='assets')
app.secret_key = 'OratorIQ_secret_key'

#Login page
@app.route('/',methods=['GET','POST'])
def index():
    error_message = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users_collection.find_one({'email': email})
        if user:
            if user['password'] == password:
                session['user'] = email
                return redirect(url_for('home'))
            else:
                error_message = 'Incorrect password'
        else:
            error_message = 'User not found'
    return render_template('Login.html', error_message=error_message)

#registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = str(uuid.uuid4())  # Generate a unique ID for the user
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        contact = request.form['contact']
        profession = request.form['profession']
        domain = request.form['domain']
        password = request.form['password']

        user = {
            'user_id': user_id,
            'first_name': first_name,
            'last_name': last_name,
            'email': email,
            'contact': contact,
            'profession': profession,
            'domain': domain,
            'password': password,
            'about': '',
            'skills': ''
        }
        result = users_collection.insert_one(user)
        if result.inserted_id:
            session['user'] = email
            return redirect(url_for('index'))
        else:
            return 'Failed to register user'
    return render_template('register.html')

#home page
@app.route('/home')
def home():
    # Redirect to the main page (root URL '/')
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('homepage.html')

# profile page
@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Update 'About' and 'Skills' fields for the logged-in user
        about = request.form.get('about')
        skills = request.form.get('skills')
        user_email = session['user']
        users_collection.update_one({'email': user_email}, {'$set': {'about': about, 'skills': skills}})
        return redirect(url_for('profile'))

    # Fetch user details from the database
    user = users_collection.find_one({'email': session['user']})
    
    user_id = get_logged_in_user_id()  # Function to get logged-in user's ID
    
    user_id = user['user_id']
    
    # Fetch all reports for the current user from the reports collection
    user_reports = fetch_user_reports(user_id)

    return render_template('profile.html', user=user, user_reports=user_reports)

#Services page
@app.route('/services')
def services():
    return render_template('services.html')

#Speech instruction page
@app.route('/services/speech-instruction')
def speech():
    return render_template('speech-instructions.html')

#Speech text page
@app.route('/services/speech-text')
def speech_text():
    # genrate the ticket number
    ticket_number = generate_ticket_number()
    session['ticket']=ticket_number
    return render_template('speech-text.html')

#speech recording page
@app.route('/services/speech_analysis')
def speech_analysis():
    return render_template('Speech.html')

#report page
@app.route('/services/speech_analysis/report')
def report():
    # Retrieve analysis data
    data,score = analysis()
    
    # Generate a ticket number
    ticket_number = session.get('ticket')
    if ticket_number is None:
        return redirect(url_for('speech_text'))
    
    # Retrieve user_id from session
    user_email = session.get('user')
    user = users_collection.find_one({'email': user_email})
    if user is None:
        return redirect(url_for('index'))
    
    user_id = user['user_id']
    
    # Create a report document
    report = {
        'ticket_number': ticket_number,
        'user_id': user_id,
        'type': 'Speech',
        'transcription': data['transcription'],
        'corrected_text': data['corrected_text'],
        'grammar_score': data['grammar_score'],
        'speech_rate': data['speech_rate'],
        'tone': data['tone'],
        'sentiment_score': data['sentiment_score'],
        'pauses_count': data['pauses_count'],
        'word_count': data['word_count'],
        'word_freq': data['word_freq'],
        'similarity_score': data['score'],
        'overall_rating': data['overall_rating'],
        'timestamp': datetime.now()
    }
    
    # Insert the report into the database
    reports_collection.insert_one(report)
    
    # Render the report template with data
    return render_template('report.html', data=data, report=report, user=user,score=json.dumps(score))

#Choose the type of interview
@app.route('/services/interview')
def interview():
    return render_template('Interview.html')

#technical Interview
@app.route('/services/interview/technical')
def technical():
    # genrate the ticket number
    ticket_number = generate_ticket_number()
    session['ticket']=ticket_number
    
    return render_template('technical.html')

#technical instruction page
@app.route('/services/Tinterview-instruction')
def Tinterview_instruction():
    return render_template('Tinterview-instruction.html')

#Interview page
@app.route('/services/interview/technical/session')
def sess():
    return render_template('TI.html')

#HR interview instruction page
@app.route('/services/Hinterview-instruction')
def Hinterview_instruction():
    return render_template('Hinterview-instruction.html')

#HR interview page
@app.route('/services/interview/HR')
def HR():
    ticket_number = generate_ticket_number()
    session['ticket']=ticket_number
    return render_template('HR.html')

#Interview report page
@app.route('/services/interview/report')
def interview_report():
    request_path=request.referrer

    if(request_path=='http://localhost:5000/services/interview/technical/session'):
        data=answer_analysis()
        report_type='Technical Interview'
    elif(request_path=='http://localhost:5000/services/interview/HR'):
        data=hr_analysis()
        report_type='HR Interview'
        
    if 'user_answer' in session:
        del session['user_answer']

    if 'my_list' in session:
        del session['my_list']

    if 'dataset_answer' in session:
        del session['dataset_answer']

    if 'questions' in session:
        del session['questions']
    
    #att_time, tot_time = generate_frames()[:2]

    ticket_number = session.get('ticket')
    if ticket_number is None:
        return redirect(url_for('interview'))
    
    user_email = session.get('user')
    user = users_collection.find_one({'email': user_email})
    if user is None:
        return redirect(url_for('index'))
    
    user_id = user['user_id']
    
    # Create a report document
    report = {
        'ticket_number': ticket_number,
        'user_id': user_id,
        'timestamp': datetime.now(),
        'type': report_type,
        'questions': data.get('questions', []),
        'user_answers': data.get('user_answer', []),
        'corrected_answers': data.get('corrected_text', []),
        'grammar_scores': data.get('grammar_score', []),
        'rating': data.get('overall_rating', []),
        'overall_rating': data['overall_rating'],
    }
    
    report_id = reports_collection.insert_one(report).inserted_id

    # Retrieve the inserted report
    inserted_report = reports_collection.find_one({'_id': ObjectId(report_id)})

    return render_template('interview-report.html', data=data, report=inserted_report, user=user, report_type=report_type)

# Redirect to the main page
@app.route('/redirect_to_main')
def redirect_to_main():
    if 'user_answer' in session:
        del session['user_answer']

    if 'my_list' in session:
        del session['my_list']

    if 'dataset_answer' in session:
        del session['dataset_answer']

    if 'questions' in session:
        del session['questions']
    # Redirect to the main page (root URL '/')
    return redirect(url_for('home'))

#logout code
@app.route('/logout')
def logout():
    session.clear()
    return  redirect(url_for('index')) 

# Check if the user is logged in by verifying if their email is stored in the session
def get_logged_in_user_id():
    if 'user' in session:
        user_email = session['user']
        user = users_collection.find_one({'email': user_email})
        if user:
            return user['user_id']
    return None

#To get questions from the list
@app.route('/get_question')
def get_question():
    request_path=request.referrer
    question=[]
    answers=[]

    if(request_path=='http://localhost:5000/services/interview/technical/session'):
        mylist=session.get('my_list',[])
        for i in mylist:
            question+= demo.question(i)[0]
            answers+=demo.question(i)[1]
        session['dataset_answer']=answers
        session['questions']=question
    elif(request_path=='http://localhost:5000/services/interview/HR'):
        question=demo.ask_question()
        session['questions']=question

    return jsonify({"question":question})

#upload the transcript of the speech
@app.route('/uploadTranscript', methods=['POST'])
def uploadTranscript():
    data = request.json
    session['user_answer']=data.get('transcriptArray')
    return jsonify({'status': 'success', 'message': 'Text saved successfully'})

#Technical answer analysis
def answer_analysis():
    similarity=[]
    grammar_score=[]
    corrected_text=[]
    overall_rating=[]
    user_answer=session.get('user_answer',[])
    dataset_answer=session.get('dataset_answer',[])
    question=session.get( 'questions', [])
    for i, questions in enumerate(question):
            print('I:',i)
            answer = next((item for item in user_answer if item['key'] == i), None)
            if answer:
                    similarity_score=demo.calculate_similarity(dataset_answer[i],user_answer[i]['value'])
                    similarity.insert(i,int(similarity_score*100))
                    grammar=au.correct_spoken_grammar(user_answer[i]['value'])
                    corrected_text.insert(i,grammar[0])
                    grammar_score.insert(i,int(grammar[1]*100))
                    overall_rating.insert(i,int(grammar[1]+similarity_score)*100/2)
            else:
                user_answer.insert(i,{'key':i, 'value':'No answer given.'})
                similarity.insert(i,0)
                corrected_text.insert(i,'')
                grammar_score.insert(i,0)
                overall_rating.insert(i,0)
    avg_similarity=statistics.mean(similarity)
    avg_rating=statistics.mean(overall_rating)
    data={'questions':question,
        'user_answer': user_answer,
          'corrected_text':corrected_text, 
          'grammar_score':grammar_score,
          'similarity':similarity,
          'avg_accuracy': avg_similarity,
          'rating': overall_rating,
          'overall_rating':avg_rating,
          'type': 'Technical Interview'
        }
    return data

#HR answer analysis
def hr_analysis():
    grammar_score=[]
    corrected_text=[]
    user_answer=session.get('user_answer',[])
    question=session.get( 'questions', [])
    for i, questions in enumerate(question):
            print('I:',i)
            answer = next((item for item in user_answer if item['key'] == i), None)
            if answer:
                        grammar=au.correct_spoken_grammar(user_answer[i]['value'])
                        corrected_text.insert(i,grammar[0])
                        grammar_score.insert(i,int(grammar[1]*100))
            else:
                        user_answer.insert(i,{'key':i,'value':'No answer given.'})
                        corrected_text.insert(i,'')
                        grammar_score.insert(i,0)
    avg_rating=statistics.mean(grammar_score)
    data={'questions':question,
        'user_answer': user_answer,
          'corrected_text':corrected_text, 
          'grammar_score':grammar_score,
          'avg_rating': round(avg_rating,2),
           'overall_rating': avg_rating,
          'type': 'HR Interview'
        }
    return data

#list of technical domains selected
@app.route('/process_list', methods=['POST'])
def process_list():
    try:
        # Get the list from the JSON request
        data = request.get_json()
        session['my_list'] = data.get('myList', [])
        return redirect(url_for('sess'))
    except Exception as e:
        return jsonify({'error': str(e)})

#Save the audio file
@app.route('/upload', methods=['POST']) 
def upload():
    audio_file = request.files['audio_data']
    ticket_number=session.get('ticket')
    if ticket_number is None:
        # You may choose to raise an error or redirect to an error page
        return redirect(url_for('speech_text'))
    wav_path = f'uploads/{ticket_number}.wav'
    audio_file.save(wav_path)

    return "Audio uploaded susccesfully"

#save the text file
@app.route('/save_text', methods=['POST'])
def save_text():
    data = request.json
    text_content = data.get('text_content', '')
    ticket_number=session.get('ticket')
    if ticket_number is None:
        # You may choose to raise an error or redirect to an error page
        return redirect(url_for('speech_text'))
    # Save the text to a file or perform other processing as needed
    with open(f'uploads/{ticket_number}.txt', 'w') as file:
        file.write(text_content)

    return jsonify({'status': 'success', 'message': 'Text saved successfully'})

#Speech analysis code
def analysis():
    ticket_number=session.get('ticket')
    if ticket_number is None:
        # You may choose to raise an error or redirect to an error page
        return redirect(url_for('speech_text'))
    wav_path = f'uploads/{ticket_number}.wav'
    txt_path = f'uploads/{ticket_number}.txt'
    transcripted_text=au.get_text(wav_path)
    grammar=au.correct_spoken_grammar(transcripted_text)
    corrected_text=grammar[0]
    grammar_score=grammar[1]
    speech_rate=au.get_speech_rate(transcripted_text,wav_path)
    speech_rate_status=speech_rate[1]
    speech_rate_score=speech_rate[0]
    similarity=au.similarity(transcripted_text,txt_path)
    similarity_score=similarity[0]
    similarity_status=similarity[1]
    sentiment=au.tone_analysis(transcripted_text)
    tone=sentiment[0]
    sentiment_score=sentiment[1]
    pauses=au.detect_pauses(wav_path)
    number_of_pauses=pauses[0]
    pauses_status=pauses[1]
    speech=au.analyze_speech(transcripted_text)
    word_count=speech[0]
    word_freq=speech[1]
    frequent={}
    for word, freq in word_freq.most_common(10):
        frequent[word]=freq
    overall_rating=(grammar_score+similarity_score)/2

    score=[int(grammar_score*100),int(sentiment_score*100),int(similarity_score*100),int(overall_rating*100)]

    data={'ticket_number':ticket_number,
          'date':datetime.now,
        'transcription': transcripted_text,
          'corrected_text':corrected_text, 
          'grammar_score':int(grammar_score*100),
          'speech_rate':speech_rate_score,
          'speech_rate_status':speech_rate_status,
          'tone':tone,
          'sentiment_score':int(sentiment_score*100),
          'pauses_count': number_of_pauses,
          'pauses_status':pauses_status,
          'word_count': word_count,
          'word_freq':frequent,
          'score':int(similarity_score*100),
          'similarity_status':similarity_status,
          'overall_rating':int(overall_rating*100)
        }
    return data,score

#generating the ticket number
def generate_ticket_number():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# Fetch reports for a user
def fetch_user_reports(user_id):
    # Query the database to fetch all reports associated with the given user_id
    user_reports = reports_collection.find({'user_id': user_id})
    return list(user_reports)

#video capturing
def process_frame(frame,timer_center):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaze.refresh(frame)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    frame = gaze.annotated_frame()

    rect_color = (0, 255, 0)

    if gaze.horizontal_ratio() is not None and gaze.vertical_ratio() is not None:
            if gaze.horizontal_ratio()<0.4:
                # Set the rectangle color to red for looking left
                rect_color = (0, 0, 255)
            elif gaze.horizontal_ratio()>0.8:
                # Set the rectangle color to red for looking right
                rect_color = (0, 0, 255)
            elif gaze.vertical_ratio()<0.4:
                # Set the rectangle color to red for looking top
                rect_color = (0, 0, 255)
            elif gaze.vertical_ratio()>0.8:
                # Set the rectangle color to red for looking bottom
                rect_color = (0, 0, 255)
            elif 0.4<gaze.horizontal_ratio()<0.8 and 0.4<gaze.vertical_ratio()<0.8:
                timer_center+=1
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), rect_color, 2)

    return frame

def generate_frames():
    cap = cv2.VideoCapture(0)
    count=0
    timer_center=0
    start=time.time()
    global generate_frames_active

    try:

        while generate_frames_active:  # Loop while flag is True
            ret, frame = cap.read()

            if not ret:
                logging.error("Failed to read frame from camera")
                break
            processed_frame = process_frame(frame,timer_center)
            count+=1
            # Encode the processed frame to JPEG
            _, jpeg = cv2.imencode('.jpg', processed_frame)

            # Convert to bytes
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    except Exception as e:
        logging.exception("Exception occurred in generate_frames loop")
        raise e

    finally:
        cap.release()
        end=time.time()

        total_time=end-start
        fps=count/total_time
        att_time=format(timer_center/fps, ".2f")
        tot_time=format(end-start, ".2f")

        return att_time,tot_time

@app.route('/stop_frames')
def stop_frames():
    global generate_frames_active
    generate_frames_active = False
    return "Stopping frame generation..."

@app.route('/video_feed')
def video_feed():
    global generate_frames_active
    generate_frames_active = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
