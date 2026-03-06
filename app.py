import os
import uuid
import threading
import subprocess
import tempfile
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

def blend_audio(job_id, vocal_path, beat_path, workdir):
    try:
        JOBS[job_id]['status'] = 'analyzing'
        vocal_dur = get_duration(vocal_path)
        beat_dur = get_duration(beat_path)
        if not vocal_dur or not beat_dur:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Could not read audio files.'
            return

        target_dur = min(vocal_dur, beat_dur)
        JOBS[job_id]['status'] = 'processing'

        processed_vocal = os.path.join(workdir, 'vocal_processed.wav')
        processed_beat = os.path.join(workdir, 'beat_processed.wav')
        output_mp3 = os.path.join(workdir, 'blend_output.mp3')

        # Process vocal: high-pass 180Hz (removes original beat bass), boost presence at 2.5k
        vocal_filter = "highpass=f=180,equalizer=f=2500:t=o:w=1:g=2,dynaudnorm=p=0.9:m=10"
        r = subprocess.run([
            'ffmpeg', '-y', '-i', vocal_path, '-t', str(target_dur),
            '-af', vocal_filter, '-ar', '44100', '-ac', '2', processed_vocal
        ], capture_output=True, timeout=120)
        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Vocal processing failed: ' + r.stderr.decode()[-300:]
            return

        # Process beat: cut mids at 2.5k to make room for vocals, boost low-end
        beat_filter = "equalizer=f=2500:t=o:w=2:g=-4,equalizer=f=200:t=o:w=1:g=2,dynaudnorm=p=0.7:m=10"
        r = subprocess.run([
            'ffmpeg', '-y', '-i', beat_path, '-t', str(target_dur),
            '-af', beat_filter, '-ar', '44100', '-ac', '2', processed_beat
        ], capture_output=True, timeout=120)
        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Beat processing failed: ' + r.stderr.decode()[-300:]
            return

        JOBS[job_id]['status'] = 'mixing'

        # Mix: vocals at 100%, beat at 85% (vocals sit on top)
        r = subprocess.run([
            'ffmpeg', '-y', '-i', processed_vocal, '-i', processed_beat,
            '-filter_complex', '[0:a]volume=1.0[v];[1:a]volume=0.85[b];[v][b]amix=inputs=2:duration=shortest:normalize=0',
            '-ar', '44100', '-ac', '2', '-b:a', '320k', output_mp3
        ], capture_output=True, timeout=120)
        if r.returncode != 0:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = 'Mix failed: ' + r.stderr.decode()[-300:]
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

@app.route('/blend', methods=['POST'])
def blend():
    if 'vocal' not in request.files or 'beat' not in request.files:
        return jsonify({'error': 'Upload both tracks.'}), 400
    vocal_file = request.files['vocal']
    beat_file = request.files['beat']
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
    t = threading.Thread(target=blend_audio, args=(job_id, vocal_path, beat_path, workdir), daemon=True)
    t.start()
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'status': job['status'], 'error': job.get('error'), 'filename': job.get('filename')})

@app.route('/download/<job_id>')
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 404
    return send_file(job['file'], as_attachment=True, download_name=job['filename'], mimetype='audio/mpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
