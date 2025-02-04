# MUST USE pip install moviepy==1.0.3.
# NEWER VERSIONS OF MOVIEPY WILL NOT WORK DUE TO BUG.
from moviepy.editor import VideoFileClip
from modules.video_processor import analyze_video
import json
import base64
from collections import namedtuple

def get_video_duration(file_path):
    with VideoFileClip(file_path) as video:
        duration = video.duration  # duration in seconds
    return duration

ClipData = namedtuple('ClipData', ['directory_name', "event_title", "event_feedback"])

def get_video_clips(video_path) -> str:

    # get the duration of the video
    video_duration = get_video_duration(video_path)

    # get the binary representaiton of the video
    with open(video_path, "rb") as video_file:
        binary_data = video_file.read()
        print(type(binary_data))
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode("utf-8")

    # analyze video and get data
    data = analyze_video(base64_string, video_duration)

    # if data is not None, split the video and return the notable events
    if data is not None:
        split_video(video_path, data)
        json_data = json.loads(data)
        clip_data = json_data["notable_events"]
        print(clip_data)

        clip1_tuple = ClipData("0.mp4", clip_data[0]["event_title"], clip_data[0]["event_feedback"])
        clip2_tuple = ClipData("1.mp4", clip_data[1]["event_title"], clip_data[1]["event_feedback"])
        clip3_tuple = ClipData("2.mp4", clip_data[2]["event_title"], clip_data[2]["event_feedback"])

        return (clip1_tuple, clip2_tuple, clip3_tuple, json_data["general_feedback"])
    else:
        print("No log file detected as output from analyze_video().")


def split_video(video_path, json_data: str):
    """Splits a video into a new file given a specific range of seconds."""

    data = json.loads(json_data)
    video_number = 0

    for events in data["notable_events"]:
        start_time = events["start_timestamp"]

        end_time = events["end_timestamp"]

        video = VideoFileClip(video_path)
        clip = video.subclip(start_time, end_time)
        # old clip name: f"{video_path.split('.')[0]}_{start_time:.0f}_{end_time:.0f}.mp4"
        clip.write_videofile(f"{video_number}.mp4")
        video_number+=1

if __name__ == "__main__":
    video_path = "interview.mp4"  # Replace with your video file path

    with open("interview_data.json", "r") as f:
        json_data = f.read()

    split_video(video_path, json_data)