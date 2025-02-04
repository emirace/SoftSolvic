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

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"ngrok tunnel available at {public_url}")

    app.run(port=5000)
    