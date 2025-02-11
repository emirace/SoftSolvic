import json
import os
import boto3
import sys
import traceback
import time

from dotenv import load_dotenv
from flask import jsonify

from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from modules.memory_class import LocalChatMemory, combine_videos, get_mp4_binary, S3ChatMemory
from modules.crop_function import get_video_clips
from modules.job_web_scrape import search_page
from modules.resume_reader import extract_text_from_pdf
import modules.s3_interactor as s3_interactor


load_dotenv()

def _get_init_prompt_template():
    return """
                            You are a interviewer and you will be interviewing an applicant for a job. The job information is either given as a website or a string with the job description. 
                            If you are given a website, use the website scraping tool to retrieve the information from the website. 
                            If you are given a string with the job description, you are to take the job description as the sole source of truth for the interview. 
                            Assume nothing about the interviewee unless the information is provided to you. 

                            As an interviewer, you are to provide one starting question to the interviewee based on what the interviewee has mentioned previously. 
                            Return exactly what you would give in your first question, nothing else. 

                            You are to end the interview at anytime you feel that a interview would end normally. 
                            That includes, but not limited to, encountering red flag responses from interviewees or you feel that you have an understanding of an interviewee.
                            You can end the interview by giving "****", that is four astrix in a row. 
                            

                            Here is the job information: {job_info}
                            Here is some information about the interviewee you may reference: {interviewee_info}

                            Remember that you are the interviewer and that you are supposed to provide question. 
                          """

def _get_next_question_prompt_template(video_input, chat_history): 
    return [
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "mp4",  # Specify the video format
                            "source": {"bytes": video_input},  # Send video as bytes
                        }
                    },
                    {"text": f"""You are an interviewer and you will be interviewing an applicant for a job.
                    Here is the previous chat history between you (the interviewer) and the interviewee: {chat_history}
                    You will also be provided with a video containing the latest response from the interviewee.

                    You are to end the interview at anytime you feel that a interview would end normally. 
                    That includes, but not limited to, encountering red flag responses from interviewees or you feel that you have an understanding of an interviewee. 

                    You are to two give outputs, a transcript of the interviewee's response and your next question for the interviewee
                    You must output a JSON file with only two keys: 
                    1. "audio_transcript"
                    2. "next_question" 
                    """+
                    """Here is a template: 
                    {
                    "audio_transcript": "nice to meet you",
                    "next_question": "tell me about yourself"
                    }
                    Remember, as an interviewer, ask questions based on the job application and position the interviewee is applying for.
                """}
                ],
            }
        ] 

def _get_next_question_prompt_template_s3(bucket: str, video_file_name: str, chat_history):
    return [
                {
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": "mp4",
                                "source": {
                                    "s3Location": {
                                        "uri": "s3://convo-ai-io/test/WIN_20250206_17_47_02_Pro.mp4", 
                                        "bucketOwner": "597088029429"
                                    }
                                }
                            }
                        },
                        {
                            "text": "Provide video titles for this clip."
                        }
                    ]
                }
            ]

# [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "video": {
#                                 "format": "mp4",
#                                 "source": {
#                                     "s3Location": {
#                                         "uri": "s3://convo-ai-io/test/WIN_20250206_17_47_02_Pro.mp4",
#                                         "bucketOwner": "597088029429"
#                                     }
#                                 }
#                             }
#                         },
#                         {
#                             "text": f"""You are an interviewer and you will be interviewing an applicant for a job.
#                     Here is the previous chat history between you (the interviewer) and the interviewee: {chat_history}
#                     You will also be provided with a video containing the latest response from the interviewee.

#                     You are to end the interview at anytime you feel that a interview would end normally. 
#                     That includes, but not limited to, encountering red flag responses from interviewees or you feel that you have an understanding of an interviewee.

#                     You are to two give outputs, a transcript of the interviewee's response and your next question for the interviewee
#                     You must output a JSON file with only two keys: 
#                     1. "audio_transcript"
#                     2. "next_question" 
#                     """+
#                     """Here is a template: 
#                     {
#                     "audio_transcript": "nice to meet you",
#                     "next_question": "tell me about yourself"
#                     }
#                     Remember, as an interviewer, ask questions based on the job application and position the interviewee is applying for.
#                     You are to only output the json and nothing else. 
#                 """
#                         }
#                     ]
#                 }
#             ]

# [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "video": {
#                             "format": "mp4",  # Specify the video format
#                             "source": {
#                                 "s3Location": {"uri": "s3://test/WIN_20250206_17_47_02_Pro.mp4"}
#                                 #"s3Location": {"uri": "s3://" + bucket + "/" + video_file_name}
#                             }, 
#                         }
#                     },
#                     {"text": f"""You are an interviewer and you will be interviewing an applicant for a job.
#                     Here is the previous chat history between you (the interviewer) and the interviewee: {chat_history}
#                     You will also be provided with a video containing the latest response from the interviewee.

#                     You are to end the interview at anytime you feel that a interview would end normally. 
#                     That includes, but not limited to, encountering red flag responses from interviewees or you feel that you have an understanding of an interviewee.

#                     You are to two give outputs, a transcript of the interviewee's response and your next question for the interviewee
#                     You must output a JSON file with only two keys: 
#                     1. "audio_transcript"
#                     2. "next_question" 
#                     """+
#                     """Here is a template: 
#                     {
#                     "audio_transcript": "nice to meet you",
#                     "next_question": "tell me about yourself"
#                     }
#                     Remember, as an interviewer, ask questions based on the job application and position the interviewee is applying for.
#                     You are to only output the json and nothing else. 
#                 """}
#                 ],
#             }
#         ]


def get_job_data(website_url: str, custom_job_str: str) -> str:
    """returns the job data as a string based on both url and input job string"""
    if website_url:
        model_input = search_page(website_url)
    elif custom_job_str:
        model_input = custom_job_str
    else:
        raise ValueError("no website url or custom job str provided")
    
    return model_input

def parse_resume(interviewee_resume: bytearray) -> str:
    """returns the resuem in bytes"""
    try:
        interviewee_resume_binary = bytes(interviewee_resume)
    except:
        interviewee_resume_binary = None

    interviewee_resume = extract_text_from_pdf(interviewee_resume_binary)
    return interviewee_resume
    
def parse_video(video_input: bytearray, file_limit: int) -> bytes:
    if sys.getsizeof(video_input) > file_limit:
            raise ValueError("File too big!")

    video_input = bytes(video_input)
    return video_input

def write_video_to_file(video_data: bytes) -> None:
    time_now = int(time.time())
    output_file_path = f'response_videos/output_video{time_now}.mp4'

    with open(output_file_path, 'wb') as f:
        f.write(video_data)

def create_success_output(kwargs) -> "jsonify":
    output = {
        "statusCode": 200,
        "body": json.dumps(kwargs)
    }
    output = jsonify(output)

    return output

def create_fail_output(error: "error") -> 'jsonify':
    stack_trace = traceback.format_exc()
    print(stack_trace)
    return {
            "statusCode": 500,
            "body": json.dumps({"error": str(error)}),
            "location": stack_trace
        } 

def setup_langchain(prompt: str):
    """sets up the langchain pipeline object"""
    bedrock_model = os.getenv("BEDROCK_MODEL")
    region = os.getenv('AWS_REGION')

    model_kwargs = { 
        "max_tokens_to_sample": 512,
        "temperature": 0.0,
    }

    llm = ChatBedrock(
        region_name = region,
        model_id=bedrock_model,
        model_kwargs=model_kwargs,
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    return chain

def invoke_chain(chain, model_inputs: dict):
    response = chain.invoke(model_inputs)
    return response

def get_bedrock_response(message_list: list) -> tuple[str, str]:
    client = boto3.client("bedrock-runtime", region_name="us-east-2")
    # MODEL_ID = os.getenv("BEDROCK_MODEL")
    MODEL_ID = "us.amazon.nova-pro-v1:0"
    # Define your system prompt(s).
    system_list = [
        {
            "text": "You are an expert media analyst. When the user provides you with a video, provide 3 potential video titles"
        }
    ]
    # Define a "user" message including both the image and a text prompt.
    message_list = [
        {
            "role": "user",
            "content": [
                
                
                {
                    "video": {
                        "format": "mp4",  # Specify the video format
                        "source": {"s3Location": {
                                "uri": "s3://convo-ai-io/test/main.mp4"
                            }},  # Send video as bytes
                    }
                },
                {
                    "text": "Provide video titles for this clip."
                }
            ]
        }
    ]
    # Configure the inference parameters.
    inf_params = {"max_new_tokens": 300, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))

    # response = client.converse(modelId=MODEL_ID, messages=message_list)
    print(json.dumps(response, indent = 4))

    next_question = json.loads(response['output']['message']['content'][0]['text'])['next_question']
    audio_transcript = json.loads(response['output']['message']['content'][0]['text'])['next_question']
    # next_question = json.loads(response["ResponseMetadata"]["output"])
    # audio_transcript

    print("bedrock response: ", audio_transcript, next_question)
    return (audio_transcript, next_question)
    
def init_interview_session(website_url: str, custom_job_str: str, interviewee_records: str,
                           mode: str, session_key: str, interviewee_resume: bytearray) -> dict[str, int | str]:
    
    interviewee_records += parse_resume(interviewee_resume)
    model_input = get_job_data(website_url, custom_job_str)
    
    print(interviewee_records)
    print(model_input)

    try:
        prompt_template = _get_init_prompt_template()
        prompt = PromptTemplate.from_template(template=prompt_template)

        chain = setup_langchain(prompt)
        response = invoke_chain(chain, {"job_info": model_input, "interviewee_info": interviewee_records})
        
        chat_memory = LocalChatMemory(session_key)
        chat_memory.reset_interview()
        chat_memory.store_job_description(model_input)
        chat_memory.store_interviewer_question(response)

        output = create_success_output({"summary": response})
        return output
    
    except Exception as e:
        return create_fail_output(e)
    
def get_interview_question(video_input: bytearray, session_key: str):
    try:
        video_input = parse_video(video_input, 1000*1000*1000)
        write_video_to_file(video_input)

        memory_obj = LocalChatMemory(session_key)
        chat_history = memory_obj.get_chat_history()
        
        message_list = _get_next_question_prompt_template(video_input, chat_history)

        audio_transcript, next_question = get_bedrock_response(message_list)
        memory_obj.store_interviewee_response(audio_transcript)  
        print(chat_history)

        output = create_success_output({
                "audio_transcript": audio_transcript,
                "next_question": next_question
            })
        return output

    except Exception as e:
        return create_fail_output(e)

def get_video_analysis():
    try:
        main_video_path = combine_videos("response_videos/")
        clip1_tuple, clip2_tuple, clip3_tuple, text_analysis = get_video_clips(main_video_path)

        clip1_binary = get_mp4_binary(clip1_tuple.directory_name)
        clip2_binary = get_mp4_binary(clip2_tuple.directory_name)
        clip3_binary = get_mp4_binary(clip3_tuple.directory_name)

        output = create_success_output({
                "text_report": text_analysis,
                "video1": clip1_binary,
                "video2": clip2_binary,
                "video3": clip3_binary,
                "video1_title": clip1_tuple.event_title, 
                "video2_title": clip2_tuple.event_title,
                "video3_title": clip3_tuple.event_title,
                "video1_feedback": clip1_tuple.event_feedback,
                "video2_feedback": clip2_tuple.event_feedback,
                "video3_feedback": clip3_tuple.event_feedback
            })
        return output

    except Exception as e:
        return create_fail_output(e)
   

def get_s3_details(session_key: str):
    """creates a directory based on the session key and returns path to directory"""
    try:
        bucket = os.getenv("IO_BUCKET")
        directory = session_key + "/"
        s3_interactor.create_directory(bucket, directory)

        output = create_success_output({
                "bucket": bucket,
                "key": directory
            })
        return output
    
    except Exception as e:
        return create_fail_output(e)

def init_interview_session_s3(website_url: str, custom_job_str: str, interviewee_records: str,
                           mode: str, session_directory: str, resume_file_path: str):
    
    try:
        bucket = os.getenv("IO_BUCKET")
        resume_byte = s3_interactor.read_pdf(bucket, resume_file_path)
        interviewee_records = extract_text_from_pdf(resume_byte)

        model_input = get_job_data(website_url, custom_job_str)

        prompt_template = _get_init_prompt_template()
        prompt = PromptTemplate.from_template(template=prompt_template)

        chain = setup_langchain(prompt)
        response = invoke_chain(chain, {"job_info": model_input, "interviewee_info": interviewee_records})
        
        chat_memory = S3ChatMemory(bucket, session_directory)
        chat_memory.reset_interview()
        chat_memory.store_job_description(model_input)
        chat_memory.store_interviewer_question(response)

        output = create_success_output({"summary": response})
        return output
    except Exception as e:
        return create_fail_output(e)

def get_interview_question_s3(session_directory: str, video_directory: str):
    try:
        bucket = os.getenv("IO_BUCKET")
        memory_obj = S3ChatMemory(bucket, session_directory)
        chat_history = memory_obj.get_chat_history()

        message_list = _get_next_question_prompt_template_s3(bucket, video_directory, chat_history)
        print(message_list)

        audio_transcript, next_question = get_bedrock_response(message_list)
        memory_obj.store_interviewee_response(audio_transcript)  
        print(chat_history)
        
        output = create_success_output({
                    "audio_transcript": audio_transcript,
                    "next_question": next_question
                })
        return output
    
    except Exception as e:
        return create_fail_output(e)


if __name__ == "__main__":
    import base64
    import boto3
    import json
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-2",
    )

    MODEL_ID = "us.amazon.nova-lite-v1:0"
    # Define your system prompt(s).
    system_list = [
        {
            "text": "You are an expert media analyst. When the user provides you with a video, provide 3 potential video titles"
        }
    ]
    # Define a "user" message including both the image and a text prompt.
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "format": "mp4",
                        "source": {
                            "s3Location": {
                                "uri": "s3://convo-ai-io/test/test2.mp4", 
                                "bucketOwner": "597088029429"
                            }
                        }
                    }
                },
                {
                    "text": "Provide video titles for this clip."
                }
            ]
        }
    ]
    # Configure the inference parameters.
    inf_params = {"max_new_tokens": 300, "top_p": 0.1, "top_k": 20, "temperature": 0.3}

    native_request = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }
    # Invoke the model and extract the response body.
    response = client.invoke_model(modelId=MODEL_ID, body=json.dumps(native_request))
    model_response = json.loads(response["body"].read())
    # Pretty print the response JSON.
    print("[Full Response]")
    print(json.dumps(model_response, indent=2))
    # Print the text content for easy readability.
    content_text = model_response["output"]["message"]["content"][0]["text"]
    print("\n[Response Content Text]")
    print(content_text)
        