import { Slide } from "@mui/material";
import { useEffect, useState } from "react";
import moment from "moment";
import ChatDisplay from "./ChatDisplay";

export default function VideoHighlights({
  currentHTML,
  setCurrentHTML,
  video1,
  video2,
  video3,
  video1_title,
  video2_title,
  video3_title,
  video1_summary,
  video2_summary,
  video3_summary,
  textReport,
}) {
  const [video1Blob, setVideo1Blob] = useState(null);
  const [video2Blob, setVideo2Blob] = useState(null);
  const [video3Blob, setVideo3Blob] = useState(null);

  useEffect(() => {
    const video1URL = URL.createObjectURL(
      new Blob([new Uint8Array(video1)], { type: "video/mp4" })
    );
    const video2URL = URL.createObjectURL(
      new Blob([new Uint8Array(video2)], { type: "video/mp4" })
    );
    const video3URL = URL.createObjectURL(
      new Blob([new Uint8Array(video3)], { type: "video/mp4" })
    );

    setVideo1Blob(video1URL);
    setVideo2Blob(video2URL);
    setVideo3Blob(video3URL);
  }, []);

  console.log(textReport);

  return (
    <div className="w-screen min-h-screen flex flex-col pt-14 py-10 px-16 right-gradient">
      <Slide direction="right" in={true} mountOnEnter timeout={2000}>
        <h1 className="my-12 text-blue-900 text-4xl font-medium text-center">
          These are the highlights our AI found
        </h1>
      </Slide>
      {/* <div className="flex flex-row gap-x-10">
                <div className="flex flex-col gap-y-4 bg-white p-4 rounded-lg px-8 max-w-[450px] shadow-lg">
                    <h1 className="text-blue-800 text-2xl font-bold">{video1_title}</h1>
                    <video src={video1Blob} className="h-[280px] w-[400px] bg-gray-800 rounded-lg mt-4" autoPlay loop />
                    <p className="text-gray-800 text-center">{video1_summary}
                    </p>
                </div>

                <div className="flex flex-col gap-y-4 bg-white p-4 rounded-lg px-8 max-w-[450px] shadow-lg">
                    <h1 className="text-blue-800 text-2xl font-bold">{video2_title}</h1>
                    <video src={video2Blob} className="h-[280px] w-[400px] bg-gray-800 rounded-lg mt-4"  autoPlay loop />
                    <p className="text-gray-800 text-center">{video2_summary}
                    </p>
                </div>

                <div className="flex flex-col gap-y-4 bg-white p-4 rounded-lg px-8 max-w-[450px] shadow-lg">
                    <h1 className="text-blue-800 text-2xl font-bold">{video3_title}</h1>
                    <video src={video3Blob} className="h-[280px] w-[400px] bg-gray-800 rounded-lg mt-4"  autoPlay loop />
                    <p className="text-gray-800 text-center">{video3_summary}
                    </p>
                </div>
            </div> */}

      <div className="text-3xl text-center mb-4">
        Overall Rating: {textReport["text_report"].overall_rating}/10
      </div>
      <div className="text-gray-800 text-xl max-w-5xl justify-center mx-auto  rounded-lg shadow-lg p-4 bg-white mb-6">
        <p className="text-2xl mb-4">General feedback: </p>
        {textReport["text_report"].general_feedback}
      </div>
      <div className="text-gray-800 text-xl max-w-5xl mb-6 justify-center mx-auto  rounded-lg shadow-lg p-4 bg-white">
        {textReport["text_report"].notable_events.map((event) => (
          <div key={event?.timestamp} className="mb-4">
            <p className="text-2xl mb-2">
              Notable event at {moment(event?.timestamp).calendar()}
            </p>
            <p>{event?.event}</p>
          </div>
        ))}
      </div>
      <ChatDisplay chatHistory={textReport["chat_history"]} />
    </div>
  );
}
