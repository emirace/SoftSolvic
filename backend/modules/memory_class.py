import json
from pathlib import Path
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import numpy as np

import modules.s3_interactor as s3_interactor

class LocalChatMemory:
    def __init__(self, session_id: str):
        self._session_id = session_id

        memory_file = Path(f'chat_histories/{session_id}.json')
        self._memory_file = memory_file
        
        if memory_file.is_file():
            with open(memory_file, 'r') as file:
                self._chat_history = json.load(file)

        else:
            self._chat_history = dict()

    def get_chat_history(self):
        return self._chat_history
    
    def _save_changes(self):
        with open(self._memory_file, 'w', encoding='utf-8') as file:
            json.dump(self._chat_history, file)
    
    def store_interviewer_question(self, question: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"Interviwer: {question}"
        self._save_changes()

    def store_job_description(self, job_description: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"Job description context: {job_description}"
        self._save_changes()

    def store_interviewee_response(self, response: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"interviewee response: {response}"
        self._save_changes()

    def reset_interview(self):
        self._chat_history = dict()
        self._save_changes()

def combine_videos(directory: str):
    video_directory = Path(directory)

    clips = []
    # Iterate through video files only (assuming mp4 files)
    for video_path in video_directory.iterdir():
        print(video_path)
        if video_path.suffix.lower() == ".mp4":  # Check for mp4 extension
            clips.append(VideoFileClip(str(video_path)))

    if clips:  # Check if there are any clips to concatenate
        output_clip = concatenate_videoclips(clips, method="compose")
        output_file = "final_videos/main.mp4"
        output_clip.write_videofile(str(output_file))

        # Remove original video files after concatenation
        for video_path in video_directory.iterdir():
            if video_path.suffix.lower() == ".mp4":
                os.remove(video_path)

        print(f"Videos have been combined and saved as {output_file}")
        return output_file
    else:
        print("No video files found to combine.")

def get_mp4_binary(filepath):
    """returns the binary of the given video file and deletes the video"""
    video_filepath = Path(filepath)
    with open(video_filepath, "rb") as file:
        video_binary = file.read()
    os.remove(video_filepath)

    int8_array = np.frombuffer(video_binary, dtype=np.int8)

    return int8_array.tolist()

class S3ChatMemory:
    def __init__(self, bucket: str, session_directory: str):
        self._bucket = bucket
        file_path = session_directory + "chat_history.json"
        self._file_path = file_path

        self._chat_history = s3_interactor.read_json(bucket, self._file_path)

    def get_chat_history(self):
        return self._chat_history
    
    def _save_changes(self):
        s3_interactor.save_as_json(self._bucket, self._file_path, self._chat_history)
    
    def store_interviewer_question(self, question: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"Interviwer: {question}"
        self._save_changes()

    def store_job_description(self, job_description: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"Job description context: {job_description}"
        self._save_changes()

    def store_interviewee_response(self, response: str):
        timestamp = datetime.now().isoformat()
        self._chat_history[timestamp] = f"interviewee response: {response}"
        self._save_changes()

    def reset_interview(self):
        self._chat_history = dict()
        self._save_changes()



if __name__ == '__main__':
    # local_chat_memory('test')
    combine_videos("response_videos")