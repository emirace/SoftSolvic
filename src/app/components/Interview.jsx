import { Drawer } from "@mui/material";
import { useEffect, useState } from "react";
import { TextEffectOne } from "react-text-animate";

import Snackbar from "@mui/material/Snackbar";
import {
  ReactMediaRecorder,
  useReactMediaRecorder,
} from "react-media-recorder";

export default function Interview({
  currentHTML,
  setCurrentHTML,
  currentQuestion,
  interviewMode,
  setRecentVideoRecording,
}) {
  let { status, startRecording, stopRecording, clearBlobUrl, mediaBlobUrl } =
    useReactMediaRecorder({
      video: true,
      audio: true,
      blobPropertyBag: { type: "video/mp4" },
      mediaRecorderOptions: {
        mimeType: 'video/mp4; codecs="avc1.424028"',
      },
    });

  const [isRecording, setIsRecording] = useState(false);
  const [recruiterOpen, setRecruiterOpen] = useState(false);
  const [clicked, setClicked] = useState(false);

  function handleStartRecording() {
    setIsRecording(true);
    startRecording();
  }

  function handleStopRecording() {
    fetch(mediaBlobUrl)
      .then((response) => response.blob())
      .then((blob) => {
        console.log(blob);

        const file = new File([blob], "video.mp4", { type: "video/mp4" });

        console.log(file);

        const reader = new FileReader();
        reader.onload = () => {
          const arrayBuffer = reader.result;
          const byteArray = new Uint8Array(arrayBuffer);
          const bytes = Array.from(byteArray);
          console.log(bytes);
          if (!clicked) {
            setClicked(true);
          } else {
            setRecentVideoRecording(bytes);
          }
        };
        reader.readAsArrayBuffer(file);

        /*
            const blobContent = blob.text();          

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.controls = true;
            a.download = 'video.mp4';
            a.click()

            console.log(url)
            
            console.log(a)

            //const videoUrl = URL.createObjectURL(videoBlob);
            console.log(blobContent.then(data => {
                console.log(typeof data)
                setRecentVideoRecording(data)
            }))
            */
      });

    setIsRecording(false);
    stopRecording();
  }

  useEffect(() => {
    handleStartRecording();
    stopRecording();
    handleStartRecording();
  }, []);

  function handleDone() {
    if (isRecording) {
      handleStopRecording();
      console.log("stopped");
    }
    handleStartRecording();
  }

  function handleEndInterview() {
    handleDone();
    setCurrentHTML(8);
  }

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
        <p className="text-center text-2xl my-6 text-gray-600">
          {currentQuestion}
        </p>
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
      </div>

      <button
        onClick={handleDone}
        className="absolute flex flex-row items-center gap-x-2 bottom-16 bg-blue-800 text-white py-2 px-6 rounded-3xl hover:bg-blue-950 transition-all right-64"
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

      <button
        onClick={handleEndInterview}
        className="absolute flex flex-row items-center gap-x-2 bottom-16 bg-blue-800 text-white py-2 px-6 rounded-3xl hover:bg-blue-950 transition-all right-20"
      >
        end interview
      </button>

      <Snackbar
        open={clicked}
        autoHideDuration={6000}
        message="Failed to send video to server, please respond again!"
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      />
    </>
  );
}
