import requests
import os, time, shutil, json
import gradio as gd
from faster_whisper import WhisperModel
import uuid
from moviepy import VideoFileClip

# DIARIZATION
import subprocess
import numpy as np
import torch
from pyannote.audio import Pipeline
import gc
# DIARIZATION

from dotenv import load_dotenv
load_dotenv(override=True)

GRADIO_SERVER_NAME = os.getenv('GRADIO_SERVER_NAME', '127.0.0.1')
GRADIO_SERVER_PORT = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
GRADIO_SERVER_PATH = os.getenv('GRADIO_SERVER_PATH', '')

WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH','./models')
WHISPER_MODEL = os.getenv('WHISPER_MODEL','base')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'default')
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE','16000'))

LLM_API_URL = os.getenv('LLM_API_URL',"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions")
LLM_MODEL = os.getenv('LLM_MODEL','@cf/mistralai/mistral-small-3.1-24b-instruct')
LLM_AUTH_TOKEN = os.getenv('LLM_AUTH_TOKEN','')
LLM_ACCOUNT_ID = os.getenv('LLM_ACCOUNT_ID','')
LLM_API_URL = LLM_API_URL.replace('{account_id}',LLM_ACCOUNT_ID)
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.2'))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096'))
LLM_CONTEXT_SIZE = int(os.getenv('LLM_CONTEXT_SIZE', '4096'))

LLM_DEFAULT_PROMPT = os.getenv('LLM_DEFAULT_PROMPT', 'Haz un resumen de la ponencia. Aproximadamente 500 palabras. Incluye un titular al principio.')
LLM_SYSTEM_PROMPT = os.getenv('LLM_SYSTEM_PROMPT', 'No des las gracias. Estilo de artículo periodístico. No seas esquemático.')


LEGAL_DISCLAIMER = os.getenv('LEGAL DISCLAIMER', "Este demostrador es una Prueba de Concepto (PoC), no un producto final verificado. Los resultados arrojados por el demostrador no están verificados.")

video_extensions = tuple(os.getenv('VIDEO_EXTENSIONS', '.mp4,.mov,.mkv' ).split(','))
audio_extensions = tuple(os.getenv('AUDIO_EXTENSIONS','.mp3,.m4a').split(','))
text_extensions = tuple(os.getenv('TEXT_EXTENSIONS','.txt,.md').split(','))

summary_header = "## Resumen: \n"

# DIARIZATION

# WhisperX (acelera muy significativamente diarization) https://github.com/m-bain/whisperX/blob/main/whisperx/audio.py
def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

#https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/

def align_transcript_diarization(transcript, diarization):
    speaker_transcriptions = []

        # Find the end time of the last segment in diarization
    last_diarization_end = get_last_segment(diarization).end

    for chunk in transcript:
        chunk_start = chunk['start']
        chunk_end = chunk['end']
        segment_text = chunk['text']

        # Handle the case where chunk_end is None
        if chunk_end is None:
            # Use the end of the last diarization segment as the default end time
            chunk_end = last_diarization_end if last_diarization_end is not None else chunk_start

        # Find the best matching speaker segment
        best_match = find_best_match(diarization, chunk_start, chunk_end)
        if best_match:
            speaker = best_match[2]  # Extract the speaker label
            speaker_transcriptions.append((speaker, chunk_start, chunk_end, segment_text))

    # Merge consecutive segments of the same speaker
    speaker_transcriptions = merge_consecutive_segments(speaker_transcriptions)
    return speaker_transcriptions

def find_best_match(diarization, start_time, end_time):
    best_match = None
    max_intersection = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turn_start = turn.start
        turn_end = turn.end

        # Calculate intersection manually
        intersection_start = max(start_time, turn_start)
        intersection_end = min(end_time, turn_end)

        if intersection_start < intersection_end:
            intersection_length = intersection_end - intersection_start
            if intersection_length > max_intersection:
                max_intersection = intersection_length
                best_match = (turn_start, turn_end, speaker)

    return best_match

def merge_consecutive_segments(segments):
    merged_segments = []
    previous_segment = None

    for segment in segments:
        if previous_segment is None:
            previous_segment = segment
        else:
            if segment[0] == previous_segment[0]:
                # Merge segments of the same speaker that are consecutive
                previous_segment = (
                    previous_segment[0],
                    previous_segment[1],
                    segment[2],
                    previous_segment[3] + segment[3]
                )
            else:
                merged_segments.append(previous_segment)
                previous_segment = segment

    if previous_segment:
        merged_segments.append(previous_segment)

    return merged_segments

def get_last_segment(annotation):
    last_segment = None
    for segment in annotation.itersegments():
        last_segment = segment
    return last_segment



def files (session_id):
    summary_file = "./summaries/" + session_id + ".md"
    transcription_file = "./transcriptions/" + session_id + ".txt"
    audio_file = "./audios/" + session_id + ".mp3"
    return audio_file, transcription_file, summary_file
    

def transcribe(media_file, language, session_id):
    now = time.time()
    for f in os.listdir("./transcriptions/"):
        file_name = os.path.join("./transcriptions/",f)
        if os.stat(file_name).st_mtime < now - 1 * 86400 and f.lower().endswith(text_extensions):
            os.remove(file_name)
    for f in os.listdir("./summaries/"):
        file_name = os.path.join("./summaries/",f)
        if os.stat(file_name).st_mtime < now - 1 * 86400 and f.lower().endswith(text_extensions):
            os.remove(file_name)
    for f in os.listdir("./audios/"):
        file_name = os.path.join("./audios/",f)
        if os.stat(file_name).st_mtime < now - 1 * 86400 and f.lower().endswith(audio_extensions):
            os.remove(file_name)
    transcription_text = ""
    transcription_list = []
    summary_text = summary_header
    session_id = str(uuid.uuid4())
    audio_file, transcription_file, summary_file = files(session_id)
    if media_file:
        if media_file.lower().endswith(audio_extensions) or media_file.lower().endswith(video_extensions):
            if media_file.lower().endswith(video_extensions):
                video_clip = VideoFileClip(media_file)
                audio_clip = video_clip.audio
                audio_clip.write_audiofile(audio_file)
                audio_clip.close()
                video_clip.close()
                media_file = audio_file
            else:
                shutil.copy(media_file,audio_file)
            if language=='detectar':
                language = None             
            whisper_model = os.path.join(WHISPER_MODEL_PATH,WHISPER_MODEL)
            model = WhisperModel(whisper_model, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            segments, info = model.transcribe(media_file, beam_size=5, language=language)   
            for segment in segments:
                transcription_list.append({"start": segment.start, "end": segment.end, "text": segment.text})
                transcription_text+=("\n%s" % (segment.text))
                yield transcription_text, transcription_list, transcription_file, audio_file, session_id, gd.update(interactive=False), gd.update(interactive=False)
            del model
            gc.collect()
        if media_file.lower().endswith(text_extensions):
            with open(media_file,"r", encoding="utf-8") as f:
                transcription_text = f.read()
                audio_file = None
        with open(transcription_file, "w", encoding="utf-8") as f:
                    f.write(transcription_text)
        yield transcription_text, transcription_list, transcription_file, audio_file, session_id, gd.update(interactive=True) if audio_file else gd.update(interactive=False), gd.update(interactive=True)
    else:
        gd.Info("Cargando video, dame unos segundos y vuelve a intentarlo.")
        return transcription_text, transcription_list, transcription_file, audio_file, session_id, gd.update(interactive=False), gd.update(interactive=False)

 
def diarize(transcription_list, session_id):
    audio_file, transcription_file, summary_file = files(session_id)
    pipeline = Pipeline.from_pretrained("./models/config.yaml")
    pipeline.to(torch.device(WHISPER_DEVICE))
    transcription_text=""
    media_file_audio = load_audio(audio_file)
    diarization = pipeline({"waveform": torch.from_numpy(media_file_audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
    diarized_list = []
    diarized_list = align_transcript_diarization(transcription_list,diarization)
    transcription_text = "\n\n".join(f"{speaker}: {text.strip()}" for speaker, start, end, text in diarized_list)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    with open(transcription_file,"w",encoding="utf-8") as f:
        f.write(transcription_text)
    return transcription_text, session_id, gd.update(interactive=True), transcription_file



def summarize(transcription_text, prompt, session_id):  
    audio_file, transcription_file, summary_file = files(session_id)
    full_prompt = prompt + LLM_SYSTEM_PROMPT
    
    try:
        response = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_AUTH_TOKEN}"
            },
        json={
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                "messages": [
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": transcription_text},
                ],
            "model": LLM_MODEL,
            "stream": True,
            "options": {"temperature": LLM_TEMPERATURE,
                        "num_ctx": LLM_CONTEXT_SIZE
                        }
        },
        verify=True,
        stream = True
        )
        if response.ok:
            summary_text = summary_header
            for line in response.iter_lines():
                decoded_line = line.decode('utf-8')[5:]
                if 'chat.completion.chunk' in decoded_line:
                    chat_buufer = json.loads(decoded_line).get('choices')[0].get('delta').get('content')
                    if chat_buufer: summary_text += chat_buufer
                    yield summary_text, summary_file, session_id
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_text)
    except requests.exceptions.ConnectionError as e:
        summary_text = "### Error: Contacte con soporte"
    yield summary_text, summary_file, session_id
    


def main():
    with gd.Blocks() as demo:
        session_id = gd.State(None)
        transcription_list = gd.State(None)

        with gd.Row():
            gd.components.Markdown(value="## PoC IA: Trascripción y resumen de ponencias por IA v0.0.1beta")
        with gd.Row():
            with gd.Column():
                gd.components.Textbox(label="Legal",interactive=False,value=LEGAL_DISCLAIMER)            
                file_input= gd.components.File(label="Cargar video,audio o transcripción", type="filepath", file_types=["video", "audio","text"])
                language = gd.components.Dropdown(["es", "en", "fr", "detectar"], label="Idioma", info="Cual es el idioma de la ponencia?")
                transcribe_btn = gd.Button(value="Transcribir", variant="primary",interactive=False)
                diarize_btn = gd.Button(value="Separar ponentes", variant="primary", interactive=False)
                prompt = gd.components.Textbox(label="Prompt para el LLM:", value=LLM_DEFAULT_PROMPT)
                process_btn = gd.Button(value="Procesar", variant="primary", interactive=False)
            with gd.Column():
                output_text = gd.components.Textbox(label="Transcripción")
                download_transcription_btn = gd.components.DownloadButton(label="Descargar transcripcion", variant="primary")
                summary_text = gd.components.Markdown()
                download_summary_btn = gd.components.DownloadButton(label="Descargar resumen", variant="primary")
                download_audio_btn = gd.components.DownloadButton(label="Descargar audio", variant="primary")
        file_input.upload(fn=lambda: gd.update(interactive=True), inputs=None, outputs=transcribe_btn, api_name=False)
        file_input.clear(fn=lambda: (gd.update(interactive=False),gd.update(interactive=False),gd.update(interactive=False)), inputs=None, outputs=[transcribe_btn, diarize_btn, process_btn], api_name=False)      
        transcribe_btn.click(fn=lambda: gd.update(interactive=False), inputs=None, outputs=transcribe_btn, api_name=False).then(fn=transcribe,inputs=[file_input, language, session_id],outputs=[output_text, transcription_list, download_transcription_btn,download_audio_btn, session_id, diarize_btn, process_btn], show_progress=True, api_name=False).then(fn=lambda: gd.update(interactive=True), inputs=None, outputs=transcribe_btn, api_name=False)
        diarize_btn.click(fn=lambda: gd.update(interactive=False), inputs=None, outputs=diarize_btn,api_name=False).then(fn=diarize,inputs=[transcription_list, session_id],outputs=[output_text, session_id, process_btn, download_transcription_btn], show_progress=True, api_name=False).then(fn=lambda: gd.update(interactive=True), inputs=None, outputs=diarize_btn,api_name=False)
        process_btn.click(fn=lambda: gd.update(interactive=False), inputs=None, outputs=process_btn,api_name=False).then(fn=summarize,inputs=[output_text, prompt, session_id],outputs=[summary_text,download_summary_btn, session_id], show_progress=True, api_name=False).then(fn=lambda: gd.update(interactive=True), inputs=None, outputs=process_btn,api_name=False)
    
    demo.queue().launch(share=False, server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT,root_path=GRADIO_SERVER_PATH)


if __name__ == "__main__":
    main()