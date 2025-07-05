import requests
import os, time
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

LLM_API_URL = os.getenv('LLM_API_URL',"http://ollama:11434/api/chat")
LLM_MODEL = os.getenv('LLM_MODEL','mistral-small3.1:24b-instruct-2503-q4_K_M')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.2'))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096'))
LLM_CONTEXT_SIZE = int(os.getenv('LLM_CONTEXT_SIZE', '4096'))

LLM_DEFAULT_PROMPT = os.getenv('LLM_DEFAULT_PROMPT', 'Haz un resumen de la ponencia. Aproximadamente 500 palabras. Incluye un titular al principio.')
LLM1_SYSTEM_PROMPT = os.getenv('LLM1_SYSTEM_PROMPT', 'No des las gracias. Estilo de artículo periodístico. No seas esquemático.')
LLM2_SYSTEM_PROMPT = os.getenv('LLM2_SYSTEM_PROMPT', 'No des las gracias. Estilo de artículo periodístico. No seas esquemático.')

LEGAL_DISCLAIMER = os.getenv('LEGAL DISCLAIMER', "Este demostrador es una Prueba de Concepto (PoC), no un producto final verificado. Los resultados arrojados por el demostrador no están verificados.")

DIARIZE_ENABLED = os.getenv('DIARIZE_ENABLED','false') == 'true'

video_extensions = tuple(os.getenv('VIDEO_EXTENSIONS', '.mp4,.mov' ).split(','))
audio_extensions = tuple(os.getenv('AUDIO_EXTENSIONS','.mp3,.m4a').split(','))
text_extensions = tuple(os.getenv('TEXT_EXTENSIONS','.txt,.md').split(','))

summary_header = "## Resumen: \n"

whisper_model = os.path.join(WHISPER_MODEL_PATH,WHISPER_MODEL)
model = WhisperModel(whisper_model, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

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
##############################################################
# DIARIZATION


def summarize(media_file, prompt, language, session_id, summarize_cb, diarize_cb):
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
    summary_text = summary_header
    session_id = str(uuid.uuid4())
    summary_file = "./summaries/" + session_id + ".md"
    transcription_file = "./transcriptions/" + session_id + ".txt"
    audio_file = "./audios/" + session_id + ".mp3"
    
    first_prompt = prompt + LLM1_SYSTEM_PROMPT
    second_prompt = prompt + LLM2_SYSTEM_PROMPT
    if media_file:
        yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=False)    
        if media_file.lower().endswith(text_extensions):
            with open(media_file,"r", encoding="utf-8") as f:
                transcription_text = f.read()
                audio_file = None
                with open(transcription_file, "w", encoding="utf-8") as f:
                    f.write(transcription_text)
        if media_file.lower().endswith(video_extensions):
            gd.Info("Obteniendo audio")
            video_clip = VideoFileClip(media_file)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(audio_file)
            audio_clip.close()
            video_clip.close()
            media_file = audio_file
        if media_file.lower().endswith(audio_extensions) or media_file.lower().endswith(video_extensions):
            if language=='detectar':
                language = None
            segments, info = model.transcribe(media_file, beam_size=5, language=language)
            gd.Info("Iniciando transcripción")     
            transcription_list = []
            for segment in segments:
                transcription_list.append({"start": segment.start, "end": segment.end, "text": segment.text})
                transcription_text+=("\n%s" % (segment.text))
                yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=False)
            
            # DIARIZATION
            if diarize_cb and DIARIZE_ENABLED:
                gd.Info("Iniciando identificación de ponentes")
                pipeline = Pipeline.from_pretrained("./models/config.yaml")
                pipeline.to(torch.device(WHISPER_DEVICE))
                transcription_text=""
                media_file_audio = load_audio(media_file)
                diarization = pipeline({"waveform": torch.from_numpy(media_file_audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
                diarized_list = []
                diarized_list = align_transcript_diarization(transcription_list,diarization)
                transcription_text = "\n\n".join(f"{speaker}: {text.strip()}" for speaker, start, end, text in diarized_list)
                yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=False)
                del pipeline
                gc.collect()
                torch.cuda.empty_cache()
            # DIARIZATION

            with open(transcription_file,"w",encoding="utf-8") as f:
                f.write(transcription_text)

        yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=False)
        if summarize_cb:
            gd.Info("Creando resumen")
            try:
                response = requests.post(
                    LLM_API_URL,
                    headers={"Content-Type": "application/json"
                        },
                    json={
                        "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                        "messages": [
                        {"role": "system", "content": first_prompt},
                        {"role": "user", "content": transcription_text},
                        ],
                        "model": LLM_MODEL,
                        "stream": False,
                        "options": {"temperature": LLM_TEMPERATURE,
                                    "num_ctx": LLM_CONTEXT_SIZE
                                    }
                    },
                    verify=False
                )
                # print(response.json())
                if response.ok:
                    result = response.json()
                    summary_text = summary_header + result['message']['content']
                else:
                    summary_text = transcription_text
            except requests.exceptions.ConnectionError as e:
                summary_text = "### Error: Contacte con soporte"
            
            try:
                response = requests.post(
                LLM_API_URL,
                headers={"Content-Type": "application/json"
                    },
                json={
                    "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                        "messages": [
                        {"role": "system", "content": second_prompt},
                        {"role": "user", "content": summary_text},
                        ],
                    "model": LLM_MODEL,
                    "stream": False,
                    "options": {"temperature": LLM_TEMPERATURE,
                                "num_ctx": LLM_CONTEXT_SIZE
                                }
                },
                verify=False
                )
                if response.ok:
                    result = response.json()
                    summary_text = summary_header + result['message']['content']
                    #summary_text += result['response']
                    with open(summary_file, "w", encoding="utf-8") as f:
                        f.write(summary_text)
            except requests.exceptions.ConnectionError as e:
                summary_text = "### Error: Contacte con soporte"
        yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=True)
        gd.Info("Trabajo finalizado")
    else:
        gd.Info("Cargando video, dame unos segundos y vuelve a intentarlo.")
        yield transcription_text, summary_text, summary_file, transcription_file, audio_file, session_id, gd.update(interactive=True)

def main():
    with gd.Blocks() as demo:
        session_id = gd.State()
        with gd.Row():
            gd.components.Markdown(value="## PoC IA AEPD: Trascripción y resumen de ponencias por IA v0.0.1beta")
        with gd.Row():
            with gd.Column():
                gd.components.Textbox(label="Legal",interactive=False,value=LEGAL_DISCLAIMER)            
                file_input = gd.components.UploadButton(type="filepath",file_types=["video", "audio","text"],label="Cargar video,audio o transcripción", variant="primary", interactive=True)
                language = gd.components.Dropdown(["es", "en", "fr", "detectar"], label="Idioma", info="Cual es el idioma de la ponencia?")
                diarize_cb = gd.components.Checkbox(label="Separar ponentes", info="Requiere considerablemente más tiempo de proceso")
                summarize_cb = gd.components.Checkbox(label="Incluir resumen", value=True)
                prompt = gd.components.Textbox(label="Prompt para el LLM (Únicamente se aplica si se marca la opción 'Incluir resumen'):", value=LLM_DEFAULT_PROMPT)
                process_btn = gd.Button(value="Procesar", variant="primary")
            with gd.Column():
                output_text = gd.components.Textbox(label="Transcripción")
                download_transcription_btn = gd.components.DownloadButton(label="Descargar transcripcion", variant="primary")
                summary_text = gd.components.Markdown()
                download_summary_btn = gd.components.DownloadButton(label="Descargar resumen", variant="primary")
                download_audio_btn = gd.components.DownloadButton(label="Descargar audio", variant="primary")
        process_btn.click(fn=summarize,inputs=[file_input,prompt, language, session_id, summarize_cb, diarize_cb],outputs=[output_text,summary_text,download_summary_btn,download_transcription_btn,download_audio_btn, session_id, process_btn], show_progress=True, api_name=False)
    
    demo.queue().launch(share=False, server_name=GRADIO_SERVER_NAME, server_port=GRADIO_SERVER_PORT,root_path=GRADIO_SERVER_PATH)


if __name__ == "__main__":
    main()