from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from chatbot import chat, clear_chat_history, convert_to_egyptian, make_punctuation_aware, tts_edit, translate
from langdetect import detect
from video_processing import visual_pred, video_to_audio
from text_preprocessing import main_text
from audio_preprocessing import audio_prediction
from EEG_preprocessing import final_prediction
import os
import uuid
import time
import shutil
from pydub import AudioSegment
from glob import glob

import asyncio
import edge_tts

def tts_edit(text, lang="en", chat_voice='female', output_path="static/response.mp3"):
    voices = {
        "ar": {"female": "ar-EG-SalmaNeural", "male": "ar-EG-ShakirNeural"},
        "en": {"female": "en-US-JennyNeural", "male": "en-US-ChristopherNeural"},
        "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
        "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"}
    }
    voice = voices.get(lang, voices["en"]).get(chat_voice, voices["en"]["female"])

    async def generate_audio():
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
        await communicate.save(output_path)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(generate_audio(), loop)
        future.result()
    else:
        asyncio.run(generate_audio())

    return output_path



app = Flask(__name__)
app.secret_key = str(uuid.uuid4())
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs('static/user_voices', exist_ok=True)

exit_words = ['exit', "خروج"]

def safe_remove(filepath, retries=5, delay=0.5):
    for _ in range(retries):
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError:
            return False 
        except PermissionError:
            time.sleep(delay)
    return False



def clear_all_session_files():
    session_id = session.get('_id', 'default')

    voice_files = glob(f'static/user_voices/{session_id}_voice_*.wav')
    for f in voice_files:
        safe_remove(f)

    for f in os.listdir('static'):
        if f.endswith(('.mp3', '.wav', '.txt')):
            safe_remove(os.path.join('static', f))

    for f in os.listdir('static/videos'):
        if f.startswith(f'session{session_id}') and f.endswith(('.webm')):
            safe_remove(os.path.join('static/videos', f))

    for f in os.listdir('static/edfs'):
        if f.endswith('.edf'):
            safe_remove(os.path.join('static/edfs', f))


def delete_temp_audio():
    for file in glob("static/user_voices/*.wav"):
        safe_remove(file)
    safe_remove("static/temp_user_voice.webm")

def merge_user_voices(session_id):
    voice_dir = 'static/user_voices'
    voice_files = sorted([
        os.path.join(voice_dir, f) for f in os.listdir(voice_dir)
        if f.startswith(f'{session_id}_voice_') and f.endswith('.wav')
    ])
    if not voice_files:
        return None
    combined = AudioSegment.empty()
    for f in voice_files:
        try:
            audio = AudioSegment.from_wav(f)
            combined += audio
        except Exception as e:
            print(f"Error loading audio {f}: {e}")
    out_path = f'static/merged_user_voice_{session_id}.wav'
    try:
        combined.export(out_path, format='wav')
    except Exception as e:
        print(f"Failed to export merged audio: {e}")
        return None

    for f in voice_files:
        safe_remove(f)

    return out_path


@app.route('/')
def splash():
    return render_template('splash.html')


@app.route('/home')
def home():
    clear_chat_history()
    session.clear()
    clear_all_session_files()
    delete_temp_audio()
    return render_template('home.html')


@app.route('/start_chat', methods=['POST'])
def start_chat():
    session.clear()
    clear_all_session_files()
    session['lang'] = request.form['language']
    session['gender'] = request.form['gender']
    session['chat_voice'] = request.form['chat_voice']
    session['_id'] = str(uuid.uuid4())
    session['voice_counter'] = 0
    delete_temp_audio()
    return redirect(url_for('chat_page'))



@app.route('/chat')
def chat_page():
    if 'lang' not in session or 'gender' not in session:
        return redirect(url_for('home'))
    return render_template('chat.html')



@app.route('/generate_response', methods=['POST'])
def generate_response():
    for i in os.listdir('static'):
        if i.endswith('.mp3'):
            safe_remove(os.path.join('static', i))

    data = request.get_json()
    text = data.get('text')
    print(f"Received text: {text}")

    lang = session.get('lang', 'en')
    gender = session.get('gender', 'female')
    chat_voice = session.get('chat_voice', 'female')
    session_id = session.get('_id', 'default')

    translated_text = translate(text, lang[:2], "en") if lang != "en" else text
    print('Translating:', translated_text)

    if text.lower() != 'start recording':
        with open(f'static/session{session_id}.txt', 'a', encoding='utf-8') as file:
            file.write(f'{text}\n')

    if text.lower() == 'start recording':
        return jsonify({'ai_text': '', 'audio_url': None})

    if text.lower() not in exit_words:
        unique_id = str(uuid.uuid4())
        audio_filename = f'static/response_{unique_id}.mp3'

        ai_text = chat(text, lang[:2])
        source_lang = detect(ai_text)
        destination_lang = lang[:2]

        if source_lang != destination_lang:
            ai_text = translate(ai_text, source_lang, destination_lang)

        if lang[:2] == 'ar':
            ai_text = convert_to_egyptian(ai_text)
            ai_text = make_punctuation_aware(ai_text)

        tts_edit(ai_text, lang[:2], chat_voice, output_path=audio_filename)
        shutil.copy(audio_filename, 'static/yoka_voice.mp3')

        return jsonify({'ai_text': ai_text, 'audio_url': '/' + audio_filename + f'?v={unique_id}'})
    else:
        return jsonify({'ai_text': 'Goodbye!', 'audio_url': None})


    

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audio = request.files['audio']
        session_id = session.get('_id', 'default')

        count = session.get('voice_counter', 0) + 1
        session['voice_counter'] = count

        temp_path = 'static/temp_user_voice.webm'
        filename = f'static/user_voices/{session_id}_voice_{count}.wav'
        audio.save(temp_path)

        os.system(f"ffmpeg -y -i {temp_path} -ar 16000 -ac 1 {filename}")
        safe_remove(temp_path)

        if os.path.getsize(filename) < 10000:
            safe_remove(filename)
            return jsonify({"status": "skipped", "reason": "audio too short"}), 200
        return jsonify({"status": "saved", "filename": filename}), 200

    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500



@app.route('/upload_video', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    if video:
        session_id = session.get('_id', 'default')
        video_path = os.path.join('static/videos', f'session{session_id}.webm')
        video.save(video_path)

        print(f"video is saved {video_path}")
        print(f"there isn't video {os.path.exists(video_path)}")

        if not os.path.exists(video_path):
            return jsonify({'error'}), 500

        session['video_path'] = video_path
        return jsonify({'filename': video.filename}), 200
    return jsonify({'error'}), 400



@app.route('/upload_eeg', methods=['GET', 'POST'])
def upload_eeg():
    session_id = session.get('_id', 'default')
    video_filename = None
    session_text = ""

    if 'video_path' in session:
        video_filename = os.path.basename(session['video_path'])

    session_txt_path = f'static/session{session_id}.txt'
    if os.path.exists(session_txt_path):
        with open(session_txt_path, 'r', encoding='utf-8') as f:
            session_text = f.read()

    if request.method == 'POST':
        edf = request.files.get('eeg_file')
        if edf:
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            edf_path = os.path.join(app.config['UPLOAD_FOLDER'], edf.filename)
            edf.save(edf_path)
            session['edf_path'] = edf_path
        return redirect(url_for('processing'))

    return render_template('upload_eeg.html',
                           video_filename=video_filename,
                           session_text=session_text)



@app.route('/processing')
def processing():
    return render_template('processing.html', lang=session.get("lang", "en"))



@app.route('/analyze', methods=['POST'])
def analyze():
    session_id = session.get('_id', 'default')
    video_path = session.get('video_path')
    edf_path = session.get('edf_path')
    merged_audio_path = merge_user_voices(session_id)

    print(f"Session ID: {session_id}")
    print(f"Video path: {video_path}")
    print(f"EDF path: {edf_path}")
    print(f"Merged audio path: {merged_audio_path}")

  
    try:
        txt_path = f'static/session{session_id}.txt'
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Text file not found: {txt_path}")
        positive_prob_text, negative_prob_text = main_text(txt_path)
        print(f"Text analysis: pos={positive_prob_text}, neg={negative_prob_text}")
    except Exception as e:
        print(f"Error in text analysis: {e}")
        positive_prob_text, negative_prob_text = 0.5, 0.5

    try:
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError("Video file not found for analysis.")
        _, positive_prob_video, negative_prob_video = visual_pred(video_path)
        print(f"Video analysis: pos={positive_prob_video}, neg={negative_prob_video}")
    except Exception as e:
        print(f"Error in video analysis: {e}")
        positive_prob_video, negative_prob_video = 0.5, 0.5


    try:
        if merged_audio_path and os.path.exists(merged_audio_path):
            print(f"Audio file for prediction: {merged_audio_path}, size: {os.path.getsize(merged_audio_path)} bytes")
            print(f"User gender: {session.get('gender')}")
            result = audio_prediction(merged_audio_path, session.get("gender"))
            print(f"audio_prediction returned: {result}")
            _, positive_prob_audio, negative_prob_audio = result
        else:
            raise FileNotFoundError("Merged audio not found")
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        positive_prob_audio, negative_prob_audio = 0.5, 0.5


    if edf_path:
        try:
            _, positive_prob_eeg, negative_prob_eeg = final_prediction(edf_path)
            print(f"EEG analysis: pos={positive_prob_eeg}, neg={negative_prob_eeg}")
        except Exception as e:
            print(f"Error in EEG analysis: {e}")
            positive_prob_eeg, negative_prob_eeg = 0.5, 0.5
        results = ((negative_prob_audio + negative_prob_video + negative_prob_text + negative_prob_eeg) / 4) * 100
        session['eeg_prediction'] = float(negative_prob_eeg)
    else:
        results = ((negative_prob_audio + negative_prob_video + negative_prob_text) / 3) * 100

    session['score'] = float(results)
    session['audio_prediction'] = float(negative_prob_audio)
    session['video_prediction'] = float(negative_prob_video)
    session['text_prediction'] = float(negative_prob_text)

    return render_template('result.html',
                           score=session.get('score', 0),
                           audio_prediction=session.get('audio_prediction', 0.5),
                           video_prediction=session.get('video_prediction', 0.5),
                           text_prediction=session.get('text_prediction', 0.5),
                           eeg_prediction=session.get('eeg_prediction', 0.5 if 'edf_path' in session else None))



@app.route('/result')
def result_page():
    return render_template('result.html',
                           score=session.get('score', 0),
                           audio_prediction=session.get('audio_prediction', 0.5),
                           video_prediction=session.get('video_prediction', 0.5),
                           text_prediction=session.get('text_prediction', 0.5),
                           eeg_prediction=session.get('eeg_prediction', 0.5 if 'edf_path' in session else None))



@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    clear_chat_history()
    session.clear()
    return jsonify({'status': 'Chat cleared successfully'})



@app.route('/clear_session', methods=['POST'])
def clear_session():
    session.clear()
    return jsonify({'status': 'Session cleared successfully'})



@app.route('/clear_files', methods=['POST'])
def clear_files():
    clear_all_session_files()
    session.clear()
    return jsonify({'status': 'Files and session cleared'})


@app.route('/exit')
def exit_session():
    clear_chat_history()
    clear_all_session_files()
    delete_temp_audio()
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
