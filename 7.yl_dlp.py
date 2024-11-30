import streamlit as st
import subprocess
import yt_dlp
import torch
import math, os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import pipeline
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from dotenv import load_dotenv

# env ファイルの読み込み
load_dotenv()

HUGGING_FACE_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# YouTube動画のタイトルを取得
def download_title(video_url):
    # yt-dlpコマンドを組み立てる
    cmd = f"yt-dlp --get-title {video_url}"
    
    # yt-dlpコマンドを実行してタイトルを取得
    title = subprocess.check_output(cmd, shell=True, text=True).strip()
    
    return title
# YouTube動画の長さを取得
def download_duration(video_url):
    # yt-dlpコマンドを組み立てる
    cmd = f"yt-dlp --get-duration {video_url}"
    
    # yt-dlpコマンドを実行してタイトルを取得
    title = subprocess.check_output(cmd, shell=True, text=True).strip()
    
    return title

# YouTube動画から音声をダウンロード
def download_wav(video_url, start_time=0, end_time=10, file_num=0, speaker=None):

    def set_download_ranges(info_dict, self):
        duration_opt = [{
            'start_time': start_time,
            'end_time': end_time
        }]
        return duration_opt


    # タイトルを取得
    video_title = download_title(video_url)
    if speaker:
        video_title = video_title + "_" + speaker
    
    video_title = video_title.replace(" ", "_")

    # yt-dlpのオプションを設定
    # オプションの情報はこちら
    # https://zenn.dev/apo_zenn/articles/b21667cc637361
    options = {
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',  # 音声を抽出するポストプロセッサ
            'preferredcodec': 'wav',  # 目的の音声形式
            'preferredquality': '192',  # 目的の音声品質
        }],
        'outtmpl': f"{video_title}_{file_num+1}.%(ext)s",  # 出力ファイル名
        'download_ranges': set_download_ranges
    }

    # ダウンロード
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

    return f"{video_title}_{file_num+1}.wav"

# 文字起こしのモデルのパイプラインを作成
def transcript_model(model_id):

    print(model_id)
    # 設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {"attn_implementation": "flash_attention_2"} if torch.cuda.is_available() else {}

    # パイプラインの作成
    pipe = pipeline(
        "automatic-speech-recognition",
        torch_dtype=torch_dtype,
        model=model_id,
        device=device,
        model_kwargs=model_kwargs,
        batch_size=16,
        trust_remote_code=True,
    )

    return pipe

# 音声ファイルの文字起こし
def transcribe_audio(pipe, file_list):
    print("transcribe_files")

    generate_kwargs = {"language": "ja", "task": "transcribe"}
    transcriptions = {}

    for file_path in file_list:
        result = pipe(file_path, chunk_length_s=15, generate_kwargs=generate_kwargs)
        transcriptions[file_path] = result['text']
    return transcriptions

def transcribe_audio_v2(pipe, file_list):
    print("transcribe_files")

    transcriptions = {}

    for file_path in file_list:
        result = pipe(file_path, chunk_length_s=15)
        print(f"{result}")
        transcriptions[file_path] = result['text']
    return transcriptions

# 音声ファイルの文字起こし結果を表示
def transcribe_outputs(pipe, file_list):

    # 音声ファイルの文字起こし
    answer = transcribe_audio(pipe, file_list)

    # 結果の表示
    for file, text in answer.items():

        # 修正するかどうか
        if correction:
            st.write('=== 文字起こし結果 === \n' + text)
            messages = [
                SystemMessage(content="日本語でチャットをしてください。ハルシネーションを起こさないで。音声を文字起こししたテキストです。誤字・脱字を修正してください"),
                HumanMessage(content=text)
            ]
            llm_result = llm.invoke(messages)
            st.write('=== 修正した結果 === \n' + llm_result.content)
        else:
            st.write('=== 文字起こし結果 === \n' + text)


# 音声ファイルの文字起こし結果を表示
def transcribe_outputs_v2(pipe, file_list):

    # 音声ファイルの文字起こし
    answer = transcribe_audio_v2(pipe, file_list)

    print(answer)

    # 結果の表示
    for file, chunks in answer.items():
        
        print(chunks)

        if correction:
            st.write('=== 文字起こし結果 === \n' + chunks)
            messages = [
                SystemMessage(content="日本語でチャットをしてください。ハルシネーションを起こさないで。音声を文字起こししたテキストです。誤字・脱字を修正してください"),
                HumanMessage(content=chunks)
            ]
            llm_result = llm.invoke(messages)
            st.write('=== 修正した結果 === \n' + llm_result.content)
        else:
            st.write('=== 文字起こし結果 === \n' + chunks)

def save_title_to_file(title):
    # タイトルをテキストファイルに保存
    with open(f"{title}.txt", "w", encoding="utf-8") as file:
        file.write(title)


# 文字起こしのモデル選択
# automatic speech recognition (ASR) model
def select_asr_model():
    model = st.sidebar.selectbox(
        "Select ASR Model",
        ("kotoba-tech/kotoba-whisper-v2.1", "openai/whisper-large-v3-turbo"),
    )

    return model

# 話者を分離して文字起こしする
def pyannote_pipeline(file_list):

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="",  # ご自身のアクセストークンを入力してください
    )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    for file_path in file_list:
        # pipelineの適用
        # "audio.wav"は処理をしたいオーディオファイルの名称を入力
        print(file_path)
        with ProgressHook() as hook:
            diarization = pipeline(file_path, hook=hook)

        # 結果の出力
        # start=0.2s stop=1.5s speaker_0
        # start=1.8s stop=3.9s speaker_1
        # start=4.2s stop=5.7s speaker_0
        # ...

        speaker_segments = []
        count = 0
        previous_end_time = 0
        current_speaker = ""

        # センテンスごとに話者と時間を取得
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start_time={turn.start:.1f}s end_time={turn.end:.1f}s speaker_{speaker}")
            if count == 0:
                start_time = turn.start
                previous_end_time = turn.end
                current_speaker = speaker
            else:
                if current_speaker != speaker:
                    item: dict = {
                        "start_time": start_time,
                        "end_time": previous_end_time,
                        "speaker": current_speaker,
                    }
                    speaker_segments.append(item)
                    start_time = turn.start
                    current_speaker = speaker

                previous_end_time = turn.end

            count += 1

        item: dict = {
            "start_time": start_time,
            "end_time": previous_end_time,
            "speaker": current_speaker,
        }
        speaker_segments.append(item)

    return speaker_segments

# 文字起こししたテキストを修正する為のモデル選択  
def select_llm_model():

    if correction :
        model = st.sidebar.selectbox(
            "Select LLM Model",
            ("gpt-4o-mini", "gpt-4o", "gpt-4o-turbo"),
        )

        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
            # base_url="...",
            # organization="...",
            # other params...
        )
    else:
        llm = None

    return llm

# 話者分離モデルの選択
def select_diarization_model():
    diarization_model = st.sidebar.selectbox(
        "Select Diarization Model",
        ("kotoba-tech/kotoba-whisper-v2.2", "pyannote/speaker-diarization-3.1"),
    )

    return diarization_model

# ビデオURLの入力
def select_video_url():
    video_url = "https://www.youtube.com/watch?v=Iw28whhyQnE"  # YouTube動画のURL
    video_url = st.text_input('YouTubeのURLを入力してください:')
    return video_url

def select_correction():
    correction = st.sidebar.checkbox('文字起こしの修正を行う')
    return correction

def select_diarization():
    diarization = st.sidebar.checkbox('話者を分離して文字起こしする')
    return diarization

def select_slice_time():
    slice_time_value = st.sidebar.selectbox(
        "分割時間",
        ("2分", "5分", "10分", "15分", "30分", "60分"),
    )

    if slice_time_value == "2分":
        slice_time = 120
    elif slice_time_value == "5分":
        slice_time = 300
    elif slice_time_value == "10分":
        slice_time = 600
    elif slice_time_value == "15分":
        slice_time = 900
    elif slice_time_value == "30分":
        slice_time = 1800
    elif slice_time_value == "60分":
        slice_time = 3600

    return slice_time

def select_overlap_time():
    overlap_time_value = st.sidebar.selectbox(
        "オーバーラップ時間",
        ("10秒", "20秒", "30秒", "なし" ),
    )

    if overlap_time_value == "10秒":
        overlap_time = 10
    elif overlap_time_value == "20秒":
        overlap_time = 20
    elif overlap_time_value == "30秒":
        overlap_time = 30
    elif overlap_time_value == "なし":
        overlap_time = 0

    return overlap_time

def select_processing_time():
    processing_time_value = st.sidebar.selectbox(
        "文字起こし時間",
        ("full", "2分", "4分", "10分", "20分", "30分"),
    )

    if processing_time_value == "full":
        processing_time_value = 0
    elif processing_time_value == "2分":
        processing_time_value = 120
    elif processing_time_value == "4分":
        processing_time_value = 240
    elif processing_time_value == "10分":
        processing_time_value = 600
    elif processing_time_value == "20分":
        processing_time_value = 1200
    elif processing_time_value == "30分":
        processing_time_value = 1800

    return overlap_time

# ファイル分割数の計算
def file_num(video_url, slice_time, processing_time):

    duration = download_duration(video_url)

    duration = duration.split(':')

    duration = int(duration[0]) * 3600 + int(duration[1]) * 60 + int(duration[2])

    if processing_time == 0:
        file_num =  math.ceil(duration / slice_time) + 1
    else:
        file_num =  math.ceil(processing_time / slice_time) + 1

    return file_num


if __name__ == '__main__': # メイン処理 --- (*4)
    # UI部分の処理 
    # --------------------
    st.title('Youtube文字起こしアプリ')

    video_url = select_video_url()
    
    st.sidebar.title('Youtube音声抽出設定')
    slice_time = select_slice_time()
    overlap_time = select_overlap_time()
    processing_time = select_processing_time()

    st.sidebar.title('文字起こしモデル設定')
    # 話者分離の設定
    diarization = select_diarization()
    # 話者分離モデル選択
    if diarization:
        asr_model_id = select_diarization_model()
    else:
        # 文字起こしモデルの選択
        asr_model_id = select_asr_model()

    # 修正モデルの読み込み
    st.sidebar.title('文字起こしの補正')
    correction = select_correction()
    llm = select_llm_model()
    
    # 音声ファイルのリスト
    audio_files = []

    if st.button('文字起こしをする'):

        # 音声ファイルのダウンロード
        start_time = 0
        end_time = slice_time

        st.write(video_url)

        for i in range(file_num(video_url, slice_time, processing_time)):
            st.write("--------------------")
            st.write(f"start_time={start_time:.1f}s end_time={end_time:.1f}s")
            st.write("download_wav...")
            audio_file = download_wav(video_url, start_time, end_time, i)

            if diarization:
                st.write("話者分離")
                st.write(f"文字起こしモデル：{asr_model_id}")
                if asr_model_id == "pyannote/speaker-diarization-3.1":
                    # モデルの読み込み
                    pipe = transcript_model("openai/whisper-large-v3-turbo")

                    speaker_segments = []
                    st.write(f"話者分離する文字起こしファイル名：{audio_file}")
    
                    # 話者分離
                    speaker_segments = pyannote_pipeline([audio_file])

                    j = 0

                    # 話者分離したセグメントごとに音声ファイルを取得
                    for segment in speaker_segments:
                        segment_start = segment["start_time"] + start_time
                        segment_end = segment["end_time"] + start_time
                        st.write("--------------------")
                        st.write(f"start_time={segment_start:.1f}s end_time={segment_end:.1f}s speaker_{segment['speaker']}")
                        st.write("download_wav...")
                        diarization_file = download_wav(video_url, segment_start, segment_end, i, segment["speaker"] + "_" + str(j))

                        st.write(f"話者 文字起こしファイル名：{diarization_file} start_time={segment_start:.1f}s end_time={segment_end:.1f}s speaker={segment['speaker']}")
                        transcribe_outputs(pipe, [diarization_file])
                        audio_files.append(diarization_file)
                        j += 1
                else:
                    # モデルの読み込み
                    pipe = transcript_model(asr_model_id)

                    st.write(f"文字起こしファイル名：{audio_file}")
                    st.write("--------------------")
                    transcribe_outputs_v2(pipe, [audio_file])
                    audio_files.append(audio_file)


            else:
                # モデルの読み込み
                pipe = transcript_model(asr_model_id)

                st.write(f"文字起こしファイル名：{audio_file}")
                st.write("--------------------")
                transcribe_outputs(pipe, [audio_file])
                audio_files.append(audio_file)
        
            start_time += slice_time - overlap_time
            end_time += slice_time 


        # 一括出力
        # # 実行
        # results = transcribe_outputs(pipe, audio_files)

        # # 結果の表示
        # for file, text in results.items():
        #     print(f"ファイル名: {file}")
        #     print(f"文字起こし結果:\n{text}\n")
