import { Drawer } from "@mui/material";
import { useEffect, useState, useRef } from "react";
import { TextEffectOne } from "react-text-animate";

import Snackbar from "@mui/material/Snackbar";

export default function Interview({
  currentHTML,
  setCurrentHTML,
  currentQuestion,
  interviewMode,
  setRecentVideoRecording,
  loadingQuestion,
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recruiterOpen, setRecruiterOpen] = useState(false);
  const [clicked, setClicked] = useState(false);
  const mediaRecorder = useRef(null);
  const [audioBlobUrl, setAudioBlobUrl] = useState(null);
  const audioChunks = useRef([]);

  useEffect(() => {
    // Start the recording when the question changes
    if (currentQuestion) {
      handleStartRecording();
    }

    return () => {
      // Cleanup: stop the recording when the component is unmounted
      if (
        mediaRecorder.current &&
        mediaRecorder.current.state === "recording"
      ) {
        mediaRecorder.current.stop();
      }
    };
  }, [currentQuestion]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true, // Only capture audio
    });

    mediaRecorder.current = new MediaRecorder(stream);

    mediaRecorder.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.current.push(event.data);
      }
    };

    mediaRecorder.current.onstop = () => {
      const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
      setAudioBlobUrl(URL.createObjectURL(audioBlob));
    };

    mediaRecorder.current.start();
  };

  const handleStartRecording = () => {
    setIsRecording(true);
    startRecording();
  };

  const handleStopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
    }
    setIsRecording(false);
  };

  const handleDone = () => {
    if (isRecording) {
      handleStopRecording();
    }
    // Do something with the audio here (e.g., send it to the backend)
    if (audioBlobUrl) {
      fetch(audioBlobUrl)
        .then((response) => response.blob())
        .then((blob) => {
          const file = new File([blob], "audio.wav", { type: "audio/wav" });
          const reader = new FileReader();
          reader.onload = () => {
            const arrayBuffer = reader.result;
            const byteArray = new Uint8Array(arrayBuffer);
            const bytes = Array.from(byteArray);
            setRecentVideoRecording(bytes);
          };
          reader.readAsArrayBuffer(file);
        });
    }
  };

  const handleEndInterview = () => {
    // handleDone();
    handleStopRecording();
    setCurrentHTML(8);
  };

  return (
    <>
      <button
        className="absolute top-6 right-6"
        onClick={() => setRecruiterOpen(true)}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="size-6"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M17.982 18.725A7.488 7.488 0 0 0 12 15.75a7.488 7.488 0 0 0-5.982 2.975m11.963 0a9 9 0 1 0-11.963 0m11.963 0A8.966 8.966 0 0 1 12 21a8.966 8.966 0 0 1-5.982-2.275M15 9.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
          />
        </svg>
      </button>
      <Drawer
        anchor="right"
        open={recruiterOpen}
        onClose={() => setRecruiterOpen(false)}
        className="bg-white max-w-[400px]"
      >
        <div className="flex flex-col items-center">Test</div>
      </Drawer>

      <div className="w-screen min-h-screen flex flex-col pt-28 py-10 px-16 right-gradient items-center justify-center">
        <div className="w-full h-[75vh] overflow-y-auto p-6 bg-gray-100">
          {currentQuestion.map((question) => (
            <div
              className={`flex w-full ${
                question.type === "agent" ? "justify-start" : "justify-end"
              }`}
            >
              <p className={`text-xl max-w-2xl my-6 text-gray-600"`}>
                {question.text}
              </p>
            </div>
          ))}
        </div>
        <hr className="border-[3px] rounded-lg w-[800px] mt-4 border-blue-700" />
        <h1 className="text-7xl font-bold tracking-widest text-center text-blue-800">
          <TextEffectOne
            lineHeight={1.5}
            text="SoftSolvic"
            animateOnce
            staggerDuration={0.05}
          />
        </h1>
        <p className="text-center text-2xl my-4 text-blue-800 font-bold">
          Coffee Chat
        </p>
        {isRecording && (
          <div className="text-2xl text-blue-800">Recording...</div>
        )}
      </div>

      <div className="fixed right-5 bottom-5 flex items-center gap-6">
        {loadingQuestion ? (
          <div className="">Loading...</div>
        ) : (
          <button
            onClick={handleDone}
            className=" flex flex-row items-center gap-x-2  bg-blue-800 text-white py-2 px-6 rounded-3xl hover:bg-blue-950 transition-all "
          >
            done
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="size-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="m4.5 12.75 6 6 9-13.5"
              />
            </svg>
          </button>
        )}

        <button
          onClick={handleEndInterview}
          className=" flex flex-row items-center gap-x-2  bg-blue-800 text-white py-2 px-6 rounded-3xl hover:bg-blue-950 transition-all "
        >
          end interview
        </button>
      </div>
      <Snackbar
        open={clicked}
        autoHideDuration={6000}
        message="Failed to send video to server, please respond again!"
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      />
    </>
  );
}
