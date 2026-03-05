import os, time, tempfile, requests
from flask import Flask, request, jsonify, send_file, render_template
from pydub import AudioSegment
import urllib.request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

REPLICATE_KEY = os.environ.get('REPLICATE_KEY', '')
DEMUCS_VERSION = 'f88aedb120e625ad464e822308515da335d50888824b975a587517068301bc0b'

HEADERS = {
    'Authorization': f'Bearer {REPLICATE_KEY}',
    'Content-Type': 'application/json',
}

def upload_to_replicate(filepath, mimetype='audio/mpeg'):
    """Upload a local file to Replicate's file storage, return URL."""
    with open(filepath, 'rb') as f:
        r = requests.post(
            'https://api.replicate.com/v1/files',
            headers={'Authorization': f'Bearer {REPLICATE_KEY}', 'Content-Type': mimetype},
            data=f,
            timeout=120
        )
    r.raise_for_status()
    data = r.json()
    return data['urls']['get']

def run_demucs(audio_url):
    """Start Demucs prediction, poll until done, return stems dict."""
    r = requests.post(
        'https://api.replicate.com/v1/predictions',
        headers=HEADERS,
        json={
            'version': DEMUCS_VERSION,
            'input': {
                'audio': audio_url,
                'model': 'htdemucs',
                'output_format': 'mp3',
                'jobs': 0,
            }
        },
        timeout=30
    )
    r.raise_for_status()
    pred = r.json()
    pred_id = pred['id']

    # Poll until done
    for _ in range(150):
        time.sleep(4)
        poll = requests.get(
            f'https://api.replicate.com/v1/predictions/{pred_id}',
            headers=HEADERS,
            timeout=30
        )
        result = poll.json()
        if result['status'] == 'succeeded':
            return result['output']  # {vocals, drums, bass, other}
        if result['status'] == 'failed':
            raise Exception(f"Demucs failed: {result.get('error', 'unknown')}")

    raise Exception('Demucs timed out after 10 minutes')

def download_stem(url, dest_path):
    """Download a stem URL to a local file."""
    urllib.request.urlretrieve(url, dest_path)

def mix_blend(vocal_stems, beat_stems, bpm_ratio, output_path):
    """
    Mix clean acapella (from vocal track) over clean instrumental (from beat).
    beat_stems drums+bass+other = clean instrumental (no vocals)
    vocal_stems vocals = clean acapella
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Download stems
        acap_path = os.path.join(tmp, 'acap.mp3')
        drums_path = os.path.join(tmp, 'drums.mp3')
        bass_path = os.path.join(tmp, 'bass.mp3')
        other_path = os.path.join(tmp, 'other.mp3')

        download_stem(vocal_stems['vocals'], acap_path)
        download_stem(beat_stems['drums'], drums_path)
        download_stem(beat_stems['bass'], bass_path)
        download_stem(beat_stems['other'], other_path)

        # Load with pydub
        acap = AudioSegment.from_file(acap_path)
        drums = AudioSegment.from_file(drums_path)
        bass = AudioSegment.from_file(bass_path)
        other = AudioSegment.from_file(other_path)

        # Clean beat = drums + bass + other overlaid
        beat_len = max(len(drums), len(bass), len(other))
        clean_beat = AudioSegment.silent(duration=beat_len, frame_rate=44100)
        clean_beat = clean_beat.overlay(drums).overlay(bass).overlay(other)
        clean_beat = clean_beat - 2  # -2dB to give vocals room

        # Time-stretch acapella to match beat BPM
        # pydub doesn't do time-stretch natively, so we adjust speed via frame rate trick
        if bpm_ratio != 1.0:
            new_rate = int(acap.frame_rate * bpm_ratio)
            acap = acap._spawn(acap.raw_data, overrides={'frame_rate': new_rate})
            acap = acap.set_frame_rate(44100)

        # Boost acapella slightly
        acap = acap + 2  # +2dB

        # Overlay acapella over clean beat
        total_len = max(len(clean_beat), len(acap))
        final = AudioSegment.silent(duration=total_len, frame_rate=44100)
        final = final.overlay(clean_beat).overlay(acap)

        # Export as WAV
        final.export(output_path, format='wav')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blend', methods=['POST'])
def blend():
    try:
        if 'vocal' not in request.files or 'beat' not in request.files:
            return jsonify({'error': 'Missing vocal or beat file'}), 400

        vocal_file = request.files['vocal']
        beat_file = request.files['beat']
        bpm_ratio = float(request.form.get('bpm_ratio', 1.0))

        vocal_name = vocal_file.filename.rsplit('.', 1)[0]
        beat_name = beat_file.filename.rsplit('.', 1)[0]

        with tempfile.TemporaryDirectory() as tmp:
            # Save uploads
            vocal_path = os.path.join(tmp, 'vocal' + os.path.splitext(vocal_file.filename)[1])
            beat_path = os.path.join(tmp, 'beat' + os.path.splitext(beat_file.filename)[1])
            vocal_file.save(vocal_path)
            beat_file.save(beat_path)

            # Upload to Replicate
            vocal_url = upload_to_replicate(vocal_path)
            beat_url = upload_to_replicate(beat_path)

            # Run Demucs on both in sequence (parallel would need threading)
            vocal_stems = run_demucs(vocal_url)
            beat_stems = run_demucs(beat_url)

            # Mix
            output_path = os.path.join(tmp, 'blend.wav')
            mix_blend(vocal_stems, beat_stems, bpm_ratio, output_path)

            # Return file
            safe_name = f"{vocal_name[:20]}_x_{beat_name[:20]}_blend.wav"
            return send_file(
                output_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=safe_name
            )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return 'OK'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
