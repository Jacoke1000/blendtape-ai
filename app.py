import os, time, tempfile, requests, numpy as np
import soundfile as sf
import urllib.request
from flask import Flask, request, jsonify, send_file, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

REPLICATE_KEY = os.environ.get('REPLICATE_KEY', '')
DEMUCS_VERSION = 'f88aedb120e625ad464e822308515da335d50888824b975a587517068301bc0b'
HEADERS = {'Authorization': f'Bearer {REPLICATE_KEY}', 'Content-Type': 'application/json'}

def upload_to_replicate(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {'.mp3': 'audio/mpeg', '.wav': 'audio/wav', '.m4a': 'audio/mp4', '.flac': 'audio/flac', '.ogg': 'audio/ogg'}
    mimetype = mime_map.get(ext, 'audio/mpeg')
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        r = requests.post(
            'https://api.replicate.com/v1/files',
            headers={'Authorization': f'Bearer {REPLICATE_KEY}'},
            files={'content': (filename, f, mimetype)},
            timeout=120
        )
    if not r.ok:
        raise Exception(f'Upload failed ({r.status_code}): {r.text[:200]}')
    return r.json()['urls']['get']

def run_demucs(audio_url):
    r = requests.post('https://api.replicate.com/v1/predictions',
        headers=HEADERS,
        json={'version': DEMUCS_VERSION, 'input': {'audio': audio_url, 'model': 'htdemucs', 'output_format': 'mp3', 'jobs': 0}},
        timeout=30)
    r.raise_for_status()
    pred_id = r.json()['id']
    for _ in range(150):
        time.sleep(4)
        result = requests.get(f'https://api.replicate.com/v1/predictions/{pred_id}', headers=HEADERS, timeout=30).json()
        if result['status'] == 'succeeded':
            return result['output']
        if result['status'] == 'failed':
            raise Exception(f"Demucs failed: {result.get('error', 'unknown')}")
    raise Exception('Timed out')

def download_stem(url):
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        return f.name

def load_audio(path, target_sr=44100):
    data, sr = sf.read(path, always_2d=True)
    if sr != target_sr:
        # simple resample ratio
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        data = np.interp(np.linspace(0, len(data)-1, new_len), np.arange(len(data)), data[:,0])
        data = data.reshape(-1, 1)
        if data.shape[1] == 1:
            data = np.hstack([data, data])
    if data.shape[1] == 1:
        data = np.hstack([data, data])
    return data.astype(np.float32), target_sr

def mix_stems(stems_list, gains):
    max_len = max(s.shape[0] for s in stems_list)
    out = np.zeros((max_len, 2), dtype=np.float32)
    for stem, gain in zip(stems_list, gains):
        out[:stem.shape[0]] += stem * gain
    # normalize
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = out / peak * 0.98
    return out

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blend', methods=['POST'])
def blend():
    try:
        if 'vocal' not in request.files or 'beat' not in request.files:
            return jsonify({'error': 'Missing files'}), 400
        vocal_file = request.files['vocal']
        beat_file = request.files['beat']
        bpm_ratio = float(request.form.get('bpm_ratio', 1.0))
        vocal_name = vocal_file.filename.rsplit('.', 1)[0]
        beat_name = beat_file.filename.rsplit('.', 1)[0]

        with tempfile.TemporaryDirectory() as tmp:
            vpath = os.path.join(tmp, 'vocal' + os.path.splitext(vocal_file.filename)[1])
            bpath = os.path.join(tmp, 'beat' + os.path.splitext(beat_file.filename)[1])
            vocal_file.save(vpath)
            beat_file.save(bpath)

            vocal_url = upload_to_replicate(vpath)
            beat_url = upload_to_replicate(bpath)

            vstems = run_demucs(vocal_url)
            bstems = run_demucs(beat_url)

            # Download stems
            acap_path = download_stem(vstems['vocals'])
            drums_path = download_stem(bstems['drums'])
            bass_path = download_stem(bstems['bass'])
            other_path = download_stem(bstems['other'])

            SR = 44100
            acap, _ = load_audio(acap_path, SR)
            drums, _ = load_audio(drums_path, SR)
            bass, _ = load_audio(bass_path, SR)
            other, _ = load_audio(other_path, SR)

            # Time-stretch acapella via resampling trick
            if abs(bpm_ratio - 1.0) > 0.01:
                orig_len = acap.shape[0]
                new_len = int(orig_len / bpm_ratio)
                x_old = np.linspace(0, 1, orig_len)
                x_new = np.linspace(0, 1, new_len)
                acap = np.column_stack([
                    np.interp(x_new, x_old, acap[:, 0]),
                    np.interp(x_new, x_old, acap[:, 1])
                ]).astype(np.float32)

            clean_beat = mix_stems([drums, bass, other], [1.0, 1.0, 1.0])
            final = mix_stems([clean_beat, acap], [0.85, 1.1])

            out_path = os.path.join(tmp, 'blend.wav')
            sf.write(out_path, final, SR)

            safe_name = f"{vocal_name[:20]}_x_{beat_name[:20]}_blend.wav"
            return send_file(out_path, mimetype='audio/wav', as_attachment=True, download_name=safe_name)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return 'OK'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
