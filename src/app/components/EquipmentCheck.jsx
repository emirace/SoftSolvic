import { useState, useEffect, useRef } from "react";

import { TextEffectOne } from "react-text-animate";
// import { useReactMediaRecorder } from "react-media-recorder";

export default function EquipmentCheck({ currentHTML, setCurrentHTML }) {
  const { status, startRecording, stopRecording, mediaBlobUrl } =
    useReactMediaRecorder({ video: true });

  const [isRecording, setIsRecording] = useState(false);
  const [tested, setTested] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [tested, setTested] = useState(false);

  function handleStartRecording() {
    setIsRecording(true);
    startRecording();
  }

  function handleStopRecording() {
    setIsRecording(false);
    stopRecording();
    setTested(true);
  }

  return (
    <div className="w-screen min-h-screen flex flex-col pt-28 py-10 px-16 right-gradient items-center">
      <h1 className="text-5xl text-center text-blue-800">
        <TextEffectOne
          lineHeight={1.5}
          text="test your video and audio below!"
          animateOnce
          staggerDuration={0.05}
        />
      </h1>
      <video
        className="h-[380px] w-[500px] bg-white mt-10"
        src={mediaBlobUrl}
        autoPlay
        loop
      />
      <p className="text-md text-gray-700 my-6">
        Please ensure your video and audio devices are working properly before
        proceeding.
      </p>
      {!isRecording ? (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={handleStartRecording}
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
              d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"
            />
          </svg>
          Start Recording
        </button>
      ) : (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={handleStopRecording}
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
              d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"
            />
          </svg>
          Stop Recording
        </button>
      )}

      {tested && (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={() => setCurrentHTML(7)}
        >
          enter
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
              d="M17.25 8.25 21 12m0 0-3.75 3.75M21 12H3"
            />
          </svg>
        </button>
      )}
    </div>
  );
  return (
    <div className="w-screen min-h-screen flex flex-col pt-28 py-10 px-16 right-gradient items-center">
      <h1 className="text-5xl text-center text-blue-800">
        <TextEffectOne
          lineHeight={1.5}
          text="test your video and audio below!"
          animateOnce
          staggerDuration={0.05}
        />
      </h1>
      <video
        className="h-[380px] w-[500px] bg-white mt-10"
        src={mediaBlobUrl}
        autoPlay
        loop
      />
      <p className="text-md text-gray-700 my-6">
        Please ensure your video and audio devices are working properly before
        proceeding.
      </p>
      {!isRecording ? (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={handleStartRecording}
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
              d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"
            />
          </svg>
          Start Recording
        </button>
      ) : (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={handleStopRecording}
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
              d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z"
            />
          </svg>
          Stop Recording
        </button>
      )}

      {tested && (
        <button
          className="flex flex-row items-center gap-x-2 mt-6 px-6 py-2 bg-blue-800 text-white rounded-3xl"
          onClick={() => setCurrentHTML(7)}
        >
          enter
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
              d="M17.25 8.25 21 12m0 0-3.75 3.75M21 12H3"
            />
          </svg>
        </button>
      )}
    </div>
  );
}
