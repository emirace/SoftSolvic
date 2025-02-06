import base64
import boto3
import json
from dotenv import load_dotenv

load_dotenv()

def analyze_video(base64_string: str, max_length: int):
    # Define the system prompt

    # Initialize the Bedrock Runtime client
    client = boto3.client("bedrock-runtime", region_name="us-east-2")
    print(type(base64_string))

    # Model ID for Nova Lite
    MODEL_ID = "us.amazon.nova-lite-v1:0"

    system_list = [
        {
            "text": """You are an interviewer. You will be provided a video as a binary string, and you must \
            evaluate the interviewee and write an analysis for their interview performance. Please format the report in JSON format. \
            You must determine if the interviewee is looking at the camera throughout its duration, and if the person is fidgeting with their hands excessively, \
            making unecessary movements, or looking around erratically. Only note down significant events, and do NOT note any insignificant events that have little feedback. \
            For any notably 'bad' or 'good' behavior or hiring points in the video, please provide a timestamp forn the beginning and end of the event in seconds \
            format along with comprehensive feedback on each event that you log. Only give the top three significant events. ONLY return the JSON. Do NOT list events past the length of the video and do NOT make events consecutive, make the the timestamp occur as the event happens. Please fill out these headers: 'overall_rating', 'general_feedback', and 'notable_events', which will hold \
            all of the timestamp-specific feedback on events that you log. Please give comprehensive general feedback in 'general_feedback' regarding the video, 200 words minimum. No less. I have given an example of a log below for a poor performance. '{'start_timestamp': '1','end_timestamp': '5', 'event_title': \
            'Interviewee looked away from the camera frequently.', 'event_feedback': 'The interviewee failed to look at the camera often, and appeared to avoid eye contact with the interviewer.'} \
            only reuse the example log, and replace the field values with your own analysis. Do NOT write feedback with a timestamp past""" +  f"{max_length} seconds. Do NOT return just the example log if you did not change the fields.",
        }
    ]

    # Define the user message with the video and analysis request
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {"bytes": base64_string},
                    }
                },
                {
                    "text": """Analyze this video and evaluate the interviewee and write an analysis for their interview performance. You must determine if the interviewee is looking at the camera throughout its duration, \
                    and if the person is fidgeting with their hands excessively, making unecessary movements, or looking around erratically. Please provide a detailed analysis of the video and \
                    and please provide a timestamp from beginning to end of the event along with a short description for any notable events that occur. ONLY return the JSON. Do NOT list events past the length of the video and do NOT make events consecutive, make the the timestamp occur as the event happens. For any notably 'bad' or \
                    'good' behavior or hiring points in the video, please provide a timestamp in seconds along with comprehensive feedback on each event that you log. \
                    Only note down significant events, and do NOT note any insignificant events that have little feedback. Only give the top three significant events. Please fill out these headers: 'overall_rating', 'general_feedback', and 'notable_events', which will hold \
                    all of the timestamp-specific feedback on events that you log. Please give comprehensive general feedback in 'general_feedback' regarding the video, 200 words minimum. No less. I have given an example of a log below for a poor performance. '{'start_timestamp': '1','end_timestamp': '5', 'event_title': \
                    'Interviewee looked away from the camera frequently.', 'event_feedback': 'The interviewee failed to look at the camera often, and appeared to avoid eye contact with the interviewer.'} \
                    only reuse the example log, and replace the field values with your own analysis. Do NOT write feedback with a timestamp past""" + f"{max_length} seconds. Do NOT return just the example log if you did not change the fields.",
                },
            ],
        }
    ]

    # Set inference parameters
    inf_params = {"max_new_tokens": 400, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

    # Create the request payload
    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    # Invoke the model
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())

    # , input=json.dumps(native_request)
    # Extract and print the analysis result
    content_text = model_response["output"]["message"]["content"][0]["text"]
    return content_text

if __name__ == "__main__":
    # Video File Path
    VIDEO_PATH = "interview.mp4"

    # Read and encode the video file
    with open(VIDEO_PATH, "rb") as video_file:
        binary_data = video_file.read()
        print(type(binary_data))
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode("utf-8")
    
        analysis = analyze_video(base64_string)