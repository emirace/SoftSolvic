import { Slide } from "@mui/material";
import { useEffect, useState } from "react";
import InterviewTypeCard from "./InterviewType";

export default function MeetingSelection({
  currentHTML,
  setCurrentHTML,
  interviewMode,
  setInterviewMode,
}) {
  const descriptions = [
    "Technical Interviews focus on the hard skills needed for the job role! This could include specific skills, certifications, and practice exercises.",
    "Coffee Chats focus on the soft skills needed for the job role! This could include communication, leadership, and problem-solving skills.",
    "Mixed Interviews focus on a combination of both hard and soft skills needed for the job role! This could include both specific skills and communication skills.",
  ];

  const [hoverDescription, setHoverDescription] = useState(null);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    if (selected == null) {
      setInterviewMode("");
    } else if (selected == 0) {
      setInterviewMode("Technical Interview");
    } else if (selected == 1) {
      setInterviewMode("Coffee Chat");
    } else if (selected == 2) {
      setInterviewMode("Mixed Interview");
    }
  }, [selected]);

  return (
    <div className="w-screen min-h-screen flex flex-col py-10 px-6 sm:px-16 right-gradient">
      <Slide direction="right" in={true} mountOnEnter timeout={1000}>
        <button className="hover:underline flex flex-row items-center text-lg sm:text-xl gap-x-4">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="size-5 sm:size-6"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M9 15 3 9m0 0 6-6M3 9h12a6 6 0 0 1 0 12h-3"
            />
          </svg>
          edit information
        </button>
      </Slide>

      <Slide direction="right" in={true} mountOnEnter timeout={2000}>
        <h1 className="my-8 sm:my-12 text-blue-900 text-2xl sm:text-4xl text-center sm:text-left">
          finally, let's choose a meeting type
        </h1>
      </Slide>

      <div className="flex flex-col sm:flex-row gap-y-8 sm:gap-x-32 px-6 sm:px-32 mt-10 sm:mt-20 justify-center items-center">
        {[
          {
            title: "Technical Interview",
            icon: "M9 17.25v1.007a3 3 0 0 1-.879 2.122L7.5 21h9l-.621-.621A3 3 0 0 1 15 18.257V17.25m6-12V15a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 15V5.25m18 0A2.25 2.25 0 0 0 18.75 3H5.25A2.25 2.25 0 0 0 3 5.25m18 0V12a2.25 2.25 0 0 1-2.25 2.25H5.25A2.25 2.25 0 0 1 3 12V5.25",
          },
          {
            title: "Coffee Chat",
            icon: "M8.625 9.75a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H8.25m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0H12m4.125 0a.375.375 0 1 1-.75 0 .375.375 0 0 1 .75 0Zm0 0h-.375m-13.5 3.01c0 1.6 1.123 2.994 2.707 3.227 1.087.16 2.185.283 3.293.369V21l4.184-4.183a1.14 1.14 0 0 1 .778-.332 48.294 48.294 0 0 0 5.83-.498c1.585-.233 2.708-1.626 2.708-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018Z",
          },
          {
            title: "Mixed Interview",
            icon: "M17.982 18.725A7.488 7.488 0 0 0 12 15.75a7.488 7.488 0 0 0-5.982 2.975m11.963 0a9 9 0 1 0-11.963 0m11.963 0A8.966 8.966 0 0 1 12 21a8.966 8.966 0 0 1-5.982-2.275M15 9.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z",
          },
        ].map((option, index) => (
          <button
            key={index}
            onClick={() => setSelected(index)}
            onMouseEnter={() => setHoverDescription(descriptions[index])}
            onMouseLeave={() => setHoverDescription(null)}
            className={`flex flex-col items-center gap-y-8 sm:gap-y-12 rounded-lg py-6 sm:py-10 px-8 sm:px-16 ${
              selected === index
                ? "bg-blue-800 text-white stroke-blue-800"
                : "bg-white border-blue-800 stroke-blue-800 text-blue-800 border-2"
            }`}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              className="size-16 sm:size-24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d={option.icon}
              />
            </svg>
            <h1 className="text-lg sm:text-2xl font-semibold text-center">
              {option.title}
            </h1>
          </button>
        ))}
      </div>

      <h2 className="text-center text-gray-400 my-4 sm:my-6">
        {hoverDescription}
      </h2>

      <button
        onClick={() => setCurrentHTML(5)}
        className="hover:bg-blue-950 text-center mx-auto my-6 sm:my-8 px-6 py-3 bg-blue-800 text-white rounded-3xl transition-all"
      >
        next step!
      </button>
    </div>
  );
}
