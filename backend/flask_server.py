import warnings

from flask import Flask
from flask import request
from flask_cors import CORS
from pyngrok import ngrok

import interview_api

app = Flask(__name__)
CORS(app)

@app.route("/init_interview_session", methods=["POST"])
def flask_init_interview_session() -> dict[str, int | str]:
    """flask api function to set up an interview question. returns the first question"""
    warnings.filterwarnings("ignore")

    data = request.get_json()

    website_url = data.get("website_url", "").strip()
    custom_job_str = str(data.get("custom_job_str", "")).strip()
    interviewee_records = str(data.get("interviewee_records")).strip()
    mode = data.get("mode", "").strip()
    session_key = data.get("session_key", "").strip()
    interviewee_resume = data.get("interviewee_resume", "")

    return interview_api.init_interview_session(website_url, custom_job_str, interviewee_records, mode, session_key, interviewee_resume)
    
@app.route("/get_interview_question", methods=["POST"])
def flask_get_interview_question():
    """Flask API function to get interview question based on user response. Returns the next question."""
    warnings.filterwarnings("ignore")

    data = request.get_json()
    
    video_input = data.get("user_input", "") # Base64 encoded video data
    session_key = data.get("session_key", "").strip()

    return interview_api.get_interview_question(video_input, session_key)

@app.route("/get_video_analysis", methods=["POST"])
def flask_get_video_analysis() -> dict[str, int | str]:
    """flask api function to get interview question based on user response. returns the next question"""
    warnings.filterwarnings("ignore")
    return interview_api.get_video_analysis()

@app.route("/get_s3_details", methods=["POST"])
def flask_get_s3_details() -> dict[str, int | str]:
    warnings.filterwarnings("ignore")
    data = request.get_json()
    session_key = str(data.get("session_key", ""))

    return interview_api.get_s3_details(session_key)

@app.route("/init_interview_session_s3", methods=["POST"])
def flask_init_interview_session_s3() -> dict[str, int | str]:
    """flask api function to set up an interview question. returns the first question"""
    warnings.filterwarnings("ignore")

    data = request.get_json()

    website_url = data.get("website_url", "").strip()
    custom_job_str = str(data.get("custom_job_str", "")).strip()
    interviewee_records = str(data.get("interviewee_records")).strip()
    mode = data.get("mode", "").strip()
    session_directory = data.get("session_directory", "").strip()
    resume_file_path = data.get("resume_file_path", "")

    return interview_api.init_interview_session_s3(website_url, custom_job_str, interviewee_records, mode, session_directory, resume_file_path)

@app.route("/get_interview_question_s3", methods=["POST"])
def flask_get_interview_question_s3():
    warnings.filterwarnings("ignore")
    data = request.get_json()

    session_directory = str(data.get("session_directory", ""))
    video_directory = str(data.get("video_directory", ""))

    return interview_api.get_interview_question_s3(session_directory, video_directory)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"ngrok tunnel available at {public_url}")

    app.run(port=5000)
    