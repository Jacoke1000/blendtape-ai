import os
import uuid
import threading
import subprocess
import tempfile
import requests
import json
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)
JOBS = {}
MAX_FILE_MB = 30

def get_duration(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=15)
        return float(result.stdout.strip())
    except:
        return None

def get_bpm(path):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream_tags=BPM',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=10)
        bpm = float(result.stdout.strip())
        if 60 <= bpm <= 200:
            return bpm
    except:
        pass
    return None

def bpm_match_score(bpm1, bpm2):
    if not bpm1 or not bpm2:
        return None
    diff = min(abs(bpm1 - bpm2), abs(bpm1 - bpm2*2), abs(bpm1*2 - bpm2))
    if diff <= 2: return 100
    if diff <= 5: return 85
    if diff <= 10: return 65
    if diff <= 20: return 40
    return 20

def get_recommendations(artist_name, vocal_bpm, beat_bpm, bpm_score):
    """Call Claude API to get DJ blend tape recommendations."""
    try:
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return None

        vbpm = round(vocal_bpm) if vocal_bpm else 'unknown'
        bbpm = round(beat_bpm) if beat_bpm else 'unknown'
        score = bpm_score if bpm_score else 'unknown'

        prompt = f"""You are a legendary hip-hop DJ and blend tape curator with encyclopedic knowledge of hip-hop history.

A user uploaded vocals from "{artist_name}" at {vbpm} BPM. They picked a beat at {bbpm} BPM. BPM compatibility score: {score}/100.

Give exactly 3 creative blend tape combinations for "{artist_name}" vocals. Think cross-era, unexpected but dope — like a real DJ would suggest.

Return ONLY a JSON array, no other text:
[
  {{
    "combo": "[Artist] vocals over [Producer/Artist] beats",
    "why": "One sentence in plain music fan language, max 12 words",
    "search": "archive.org search term for the beat"
  }}
]

Rules:
- Use real well-known artist/producer names
- No BPM numbers in the "why" field — use vibe/energy/feel language only
- If score is under 50, first suggestion should fix the energy mismatch
- Make it sound like a knowledgeable DJ friend is talking, not a robot"""

        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'content-type': 'application/json'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 500,
                'messages': [{'role': 'user', 'content': prompt}]
            },
            timeout=15
        )

        if response.status_code == 200:
            text = response.json()['content'][0]['text'].strip()
            # Strip any markdown fences
            text = text.replace('```json', '').replace('```', '').strip()
            suggestions = json.loads(text)
            # Add archive.org search URLs
            for s in suggestions:
                query = s.get('search', '').replace(' ', '+')
                s['url'] = f"https://archive.org/search?query={query}+instrumental+hip+hop"
            return suggestions
    except Exception as e:
        print(f"Recommendation error: {e}")
    return None

def blend_audio(job_id, vocal_path, beat_path, workdir, loop_beat=False):
    try:
        JOBS[job_id]['status'] = 'analyzing'
        vocal_dur = get_duration(vocal_path)
        beat_dur = get_duration(beat_path)
        if not vocal_dur or not beat_dur:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Could not read audio files.'
            return

        vocal_bpm = get_bpm(vocal_path)
        beat_bpm = get_bpm(beat_path)
        bpm_score = bpm_match_score(vocal_bpm, beat_bpm)
        JOBS[job_id]['vocal_bpm'] = vocal_bpm
        JOBS[job_id]['beat_bpm'] = beat_bpm
        JOBS[job_id]['bpm_score'] = bpm_score

        JOBS[job_id]['status'] = 'processing'

        processed_vocal = os.path.join(workdir, 'vocal_processed.wav')
        processed_beat = os.path.join(workdir, 'beat_processed.wav')
        output_mp3 = os.path.join(workdir, 'blend_output.mp3')

        vocal_filter = "highpass=f=180,equalizer=f=2500:t=o:w=1:g=2,dynaudnorm=p=0.9:m=10"
        r = subprocess.run([
            'ffmpeg', '-y', '-i', vocal_path, '-t', str(vocal_dur),
            '-af', vocal_filter, '-ar', '44100', '-ac', '2', processed_vocal
        ], capture_output=True, timeout=120)
        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Vocal processing failed.'
            return

        beat_filter = "equalizer=f=2500:t=o:w=2:g=-4,equalizer=f=200:t=o:w=1:g=2,dynaudnorm=p=0.7:m=10"

        if loop_beat and beat_dur < vocal_dur:
            loops_needed = int(vocal_dur / beat_dur) + 2
            loop_list = os.path.join(workdir, 'loop_list.txt')
            with open(loop_list, 'w') as f:
                for _ in range(loops_needed):
                    f.write(f"file '{beat_path}'\n")
            r = subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', loop_list,
                '-t', str(vocal_dur),
                '-af', beat_filter, '-ar', '44100', '-ac', '2', processed_beat
            ], capture_output=True, timeout=180)
        else:
            target_dur = min(vocal_dur, beat_dur)
            r = subprocess.run([
                'ffmpeg', '-y', '-i', beat_path, '-t', str(target_dur),
                '-af', beat_filter, '-ar', '44100', '-ac', '2', processed_beat
            ], capture_output=True, timeout=120)
            if not loop_beat and beat_dur < vocal_dur:
                trimmed = os.path.join(workdir, 'vocal_trimmed.wav')
                subprocess.run(['ffmpeg', '-y', '-i', processed_vocal, '-t', str(target_dur), trimmed],
                    capture_output=True, timeout=60)
                processed_vocal = trimmed

        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Beat processing failed.'
            return

        JOBS[job_id]['status'] = 'mixing'
        r = subprocess.run([
            'ffmpeg', '-y', '-i', processed_vocal, '-i', processed_beat,
            '-filter_complex', '[0:a]volume=1.0[v];[1:a]volume=0.85[b];[v][b]amix=inputs=2:duration=shortest:normalize=0',
            '-ar', '44100', '-ac', '2', '-b:a', '320k', output_mp3
        ], capture_output=True, timeout=180)

        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Mix failed.'
            return

        JOBS[job_id]['status'] = 'done'
        JOBS[job_id]['file'] = output_mp3
        JOBS[job_id]['filename'] = 'blend_' + job_id[:8] + '.mp3'

    except subprocess.TimeoutExpired:
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = 'Timed out. Try files under 5 minutes.'
    except Exception as e:
        JOBS[job_id]['status'] = 'error'
        JOBS[job_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'vocal' not in request.files or 'beat' not in request.files:
        return jsonify({'error': 'Upload both tracks.'}), 400

    vocal_file = request.files['vocal']
    beat_file = request.files['beat']
    artist_name = request.form.get('artist_name', '').strip()

    workdir = tempfile.mkdtemp()
    vocal_path = os.path.join(workdir, 'vocal.mp3')
    beat_path = os.path.join(workdir, 'beat.mp3')
    vocal_file.save(vocal_path)
    beat_file.save(beat_path)

    vocal_dur = get_duration(vocal_path)
    beat_dur = get_duration(beat_path)
    vocal_bpm = get_bpm(vocal_path)
    beat_bpm = get_bpm(beat_path)
    bpm_score = bpm_match_score(vocal_bpm, beat_bpm)
    needs_loop = vocal_dur and beat_dur and beat_dur < vocal_dur

    recommendations = None
    if artist_name:
        recommendations = get_recommendations(artist_name, vocal_bpm, beat_bpm, bpm_score)

    return jsonify({
        'vocal_dur': round(vocal_dur) if vocal_dur else None,
        'beat_dur': round(beat_dur) if beat_dur else None,
        'vocal_bpm': round(vocal_bpm, 1) if vocal_bpm else None,
        'beat_bpm': round(beat_bpm, 1) if beat_bpm else None,
        'bpm_score': bpm_score,
        'needs_loop': needs_loop,
        'recommendations': recommendations,
        'workdir': workdir
    })

@app.route('/blend', methods=['POST'])
def blend():
    if 'vocal' not in request.files or 'beat' not in request.files:
        return jsonify({'error': 'Upload both tracks.'}), 400

    vocal_file = request.files['vocal']
    beat_file = request.files['beat']
    loop_beat = request.form.get('loop_beat', 'false').lower() == 'true'

    for f in [vocal_file, beat_file]:
        f.seek(0, 2)
        if f.tell() / (1024*1024) > MAX_FILE_MB:
            return jsonify({'error': f'Files must be under {MAX_FILE_MB}MB.'}), 400
        f.seek(0)

    job_id = str(uuid.uuid4())
    workdir = tempfile.mkdtemp()
    vocal_path = os.path.join(workdir, 'vocal.mp3')
    beat_path = os.path.join(workdir, 'beat.mp3')
    vocal_file.save(vocal_path)
    beat_file.save(beat_path)

    JOBS[job_id] = {'status': 'starting', 'file': None, 'filename': None, 'error': None}
    t = threading.Thread(target=blend_audio, args=(job_id, vocal_path, beat_path, workdir, loop_beat), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status': job['status'],
        'error': job.get('error'),
        'filename': job.get('filename'),
        'bpm_score': job.get('bpm_score'),
        'vocal_bpm': job.get('vocal_bpm'),
        'beat_bpm': job.get('beat_bpm')
    })

@app.route('/download/<job_id>')
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    return send_file(job['file'], as_attachment=True, download_name=job['filename'], mimetype='audio/mpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
