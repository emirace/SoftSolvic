import { TextEffectOne } from "react-text-animate";
import { CircularProgress } from "@mui/material";
import { useEffect } from "react";
import { useState } from "react";

import Fade from "@mui/material/Fade";

export default function ReportLoad({
  currentHTML,
  setCurrentHTML,
  textReport,
  setTextReport,
  video1,
  setVideo1,
  video2,
  setVideo2,
  video3,
  setVideo3,
  video1_title,
  setVideo1_title,
  video2_title,
  setVideo2_title,
  video3_title,
  setVideo3_title,
  video1_summary,
  setVideo1_summary,
  video2_summary,
  setVideo2_summary,
  video3_summary,
  setVideo3_summary,
  API_URL,
  secretkey,
}) {
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(API_URL + "get_video_analysis", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
        "ngrok-skip-browser-warning": "true",
      },
      body: JSON.stringify({ secretkey }),
    })
      .then((res) => res.json())
      .then((data) => {
        const json_data = JSON.parse(data.body);

        setTextReport(json_data["text_report"]);

        setVideo1(json_data["video1"]);
        setVideo2(json_data["video2"]);
        setVideo3(json_data["video3"]);

        setVideo1_title(json_data["video1_title"]);
        setVideo2_title(json_data["video2_title"]);
        setVideo3_title(json_data["video3_title"]);

        setVideo1_summary(json_data["video1_feedback"]);
        setVideo2_summary(json_data["video2_feedback"]);
        setVideo3_summary(json_data["video3_feedback"]);
      });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentHTML(currentHTML + 1);
      clearTimeout(interval);
    }, 15000);

    const interval2 = setInterval(() => {
      setLoading(true);
      clearTimeout(interval2);
    }, 3000);

    const interval3 = setInterval(() => {
      setLoading(false);
      clearTimeout(interval3);
    }, 28000);
  }, []);

  return (
    <div className="text-blue-900 text-5xl w-screen h-screen flex flex-col items-center justify-center gap-y-16 left-gradient">
      <h1 className="text-5xl animate-fadeOut28s">
        <TextEffectOne text="all done!" animateOnce staggerDuration={0.05} />
      </h1>
      <h1 className="text-5xl animate-fadeOut28s">
        <TextEffectOne
          text="generating inteview report now"
          lineHeight={1.5}
          initialDelay={1}
          animateOnce
          staggerDuration={0.075}
        />
      </h1>
      <Fade timeout={2000} in={loading}>
        <CircularProgress />
      </Fade>
    </div>
  );
}
