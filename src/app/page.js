"use client";
"use client";

import Image from "next/image";
import dynamic from "next/dynamic.js";
import { useState, useEffect } from "react";
import Introduction from "./components/Introduction.jsx";
import GetStarted from "./components/GetStarted.jsx";
import JobDesc from "./components/JobDesc.jsx";
import PersonalDesc from "./components/PersonalDesc.jsx";
import InformationLoad from "./components/InformationLoad.jsx";
import MeetingSelection from "./components/MeetingSelection.jsx";
const EquipmentCheck = dynamic(
  () => import("./components/EquipmentCheck.jsx"),
  {
    ssr: false,
  }
);
const Interview = dynamic(() => import("./components/Interview.jsx"), {
  ssr: false,
});
// import Interview from "./components/Interview.jsx";
import VideoHighlights from "./components/VideoHighlights.jsx";
import ReportLoad from "./components/ReportLoad.jsx";

export default function Home() {
  const [currentHTML, setCurrentHTML] = useState(0);

  const [jobLink, setJobLink] = useState("");
  const [formJobTitle, setFormJobTitle] = useState("");
  const [formEmployerName, setFormEmployerName] = useState("");
  const [formEmployerLocation, setFormEmployerLocation] = useState("");
  const [formJobDescription, setFormJobDescription] = useState("");
  const [formEmployerDescription, setFormEmployerDescription] = useState("");

  const [resumeName, setResumeName] = useState("Ex. John_Doe.pdf");
  const [resumeFile, setResumeFile] = useState(null);

  const [formFullName, setFormFullName] = useState("");
  const [formPersonalJobTitle, setFormPersonalJobTitle] = useState("");
  const [formDegrees, setFormDegrees] = useState([]);
  const [formSchools, setFormSchools] = useState([]);
  const [formSchoolStartDates, setFormSchoolStartDates] = useState([]);
  const [formSchoolEndDates, setFormSchoolEndDates] = useState([]);

  const [formExperienceJobTitles, setFormExperienceJobTitles] = useState([]);
  const [formExperienceStartDates, setFormExperienceStartDates] = useState([]);
  const [formExperienceEndDates, setFormExperienceEndDates] = useState([]);
  const [formExperienceDescriptions, setFormExperienceDescriptions] = useState(
    []
  );
  const [formExperienceCompanies, setFormExperienceCompanies] = useState([]);
  const [formExperienceLocations, setFormExperienceLocations] = useState([]);

  const [currentQuestion, setCurrentQuestion] = useState("");

  const [interviewMode, setInterviewMode] = useState("");
  const [recentVideoRecording, setRecentVideoRecording] = useState("");

  const [textReport, setTextReport] = useState("");
  const [video1, setVideo1] = useState("");
  const [video2, setVideo2] = useState("");
  const [video3, setVideo3] = useState("");
  const [video1_title, setVideo1_title] = useState("");
  const [video2_title, setVideo2_title] = useState("");
  const [video3_title, setVideo3_title] = useState("");
  const [video1_summary, setVideo1_summary] = useState("");
  const [video2_summary, setVideo2_summary] = useState("");
  const [video3_summary, setVideo3_summary] = useState("");

  // const API_URL = 'https://e552-128-195-96-60.ngrok-free.app/'
  const API_URL = process.env.NEXT_PUBLIC_API_URL;
  console.log("api url:" + API_URL);

  const initializeData = {
    jobLink: jobLink,
    formJobTitle: formJobTitle,
    formEmployerName: formEmployerName,
    formEmployerLocation: formEmployerLocation,
    formJobDescription: formJobDescription,
    formEmployerDescription: formEmployerDescription,

    resumeName: resumeName,
    resumeFile: resumeFile,

    formFullName: formFullName,
    formPersonalJobTitle: formPersonalJobTitle,
    formDegrees: formDegrees,
    formSchools: formSchools,
    formSchoolStartDates: formSchoolStartDates,
    formSchoolEndDates: formSchoolEndDates,

    formExperienceJobTitles: formExperienceJobTitles,
    formExperienceStartDates: formExperienceStartDates,
    formExperienceEndDates: formExperienceEndDates,
    formExperienceDescriptions: formExperienceDescriptions,
    formExperienceCompanies: formExperienceCompanies,
    formExperienceLocations: formExperienceLocations,

    interviewMode: interviewMode,
  };

  const HTML_PARTS = [
    <Introduction currentHTML={0} setCurrentHTML={setCurrentHTML} />,
    <GetStarted currentHTML={1} setCurrentHTML={setCurrentHTML} />,
    <JobDesc
      currentHTML={2}
      setCurrentHTML={setCurrentHTML}
      jobLink={jobLink}
      setJobLink={setJobLink}
      formJobTitle={formJobTitle}
      formEmployerName={formEmployerName}
      formEmployerLocation={formEmployerLocation}
      formJobDescription={formJobDescription}
      formEmployerDescription={formEmployerDescription}
      setFormJobTitle={setFormJobTitle}
      setFormEmployerName={setFormEmployerName}
      setFormEmployerLocation={setFormEmployerLocation}
      setFormJobDescription={setFormJobDescription}
      setFormEmployerDescription={setFormEmployerDescription}
    />,
    <PersonalDesc
      currentHTML={3}
      setCurrentHTML={setCurrentHTML}
      resumeName={resumeName}
      setResume={setResumeName}
      setResumeFile={setResumeFile}
      formFullName={formFullName}
      setFormFullName={setFormFullName}
      formPersonalJobTitle={formPersonalJobTitle}
      setFormPersonalJobTitle={setFormPersonalJobTitle}
      formDegrees={formDegrees}
      setFormDegrees={setFormDegrees}
      formSchools={formSchools}
      setFormSchools={setFormSchools}
      formSchoolStartDates={formSchoolStartDates}
      setFormSchoolStartDates={setFormSchoolStartDates}
      formSchoolEndDates={formSchoolEndDates}
      setFormSchoolEndDates={setFormSchoolEndDates}
      formExperienceJobTitles={formExperienceJobTitles}
      setFormExperienceJobTitles={setFormExperienceJobTitles}
      formExperienceStartDates={formExperienceStartDates}
      setFormExperienceStartDates={setFormExperienceStartDates}
      formExperienceEndDates={formExperienceEndDates}
      setFormExperienceEndDates={setFormExperienceEndDates}
      formExperienceDescriptions={formExperienceDescriptions}
      setFormExperienceDescriptions={setFormExperienceDescriptions}
      formExperienceCompanies={formExperienceCompanies}
      setFormExperienceCompanies={setFormExperienceCompanies}
      formExperienceLocations={formExperienceLocations}
      setFormExperienceLocations={setFormExperienceLocations}
    />,
    <MeetingSelection
      currentHTML={4}
      setCurrentHTML={setCurrentHTML}
      interviewMode={interviewMode}
      setInterviewMode={setInterviewMode}
    />,
    <InformationLoad currentHTML={5} setCurrentHTML={setCurrentHTML} />,
    <EquipmentCheck currentHTML={6} setCurrentHTML={setCurrentHTML} />,
    <Interview
      currentHTML={7}
      setCurrentHTML={setCurrentHTML}
      currentQuestion={currentQuestion}
      interviewMode={interviewMode}
      setRecentVideoRecording={setRecentVideoRecording}
    />,
    <ReportLoad
      currentHTML={8}
      setCurrentHTML={setCurrentHTML}
      textReport={textReport}
      video1={video1}
      video2={video2}
      video3={video3}
      setTextReport={setTextReport}
      setVideo1={setVideo1}
      setVideo2={setVideo2}
      setVideo3={setVideo3}
      video1_title={video1_title}
      video2_title={video2_title}
      video3_title={video3_title}
      setVideo1_title={setVideo1_title}
      setVideo2_title={setVideo2_title}
      setVideo3_title={setVideo3_title}
      video1_summary={video1_summary}
      video2_summary={video2_summary}
      video3_summary={video3_summary}
      setVideo1_summary={setVideo1_summary}
      setVideo2_summary={setVideo2_summary}
      setVideo3_summary={setVideo3_summary}
      API_URL={API_URL}
    />,
    <VideoHighlights
      currentHTML={9}
      setCurrentHTML={setCurrentHTML}
      video1={video1}
      video2={video2}
      video3={video3}
      video1_title={video1_title}
      video2_title={video2_title}
      video3_title={video3_title}
      video1_summary={video1_summary}
      video2_summary={video2_summary}
      video3_summary={video3_summary}
      textReport={textReport}
    />,
  ];

  console.log({
    video1: video1,
    video2: video2,
    video3: video3,

    video1_title: video1_title,
    video2_title: video2_title,
    video3_title: video3_title,

    video1_summary: video1_summary,
    video2_summary: video2_summary,
    video3_summary: video3_summary,

    textReport: textReport,
  });

  /*
  const fakeData = {
    
    formDegrees: ['Computer Science'],
    formEmployerDescription: "Best shit on the planet.",
    formEmployerLocation: "California",
    formEmployerName: "Google", 
    formExperienceCompanies: ['Microsoft'],
    formExperienceDescriptions: ['I make amazing shit.'],
    formExperienceEndDates: ["Fri Jan 24 2025 00:00:00 GMT-0800 (Pacific Standard Time)"],
    formExperienceJobTitles: ['Software Engineer'],
    formExperienceLocations: ['California'],
    formExperienceStartDates: ["Thu Jan 23 2025 00:00:00 GMT-0800 (Pacific Standard Time)"],
    formFullName: "NTCHE",
    formJobDescription: "Make better search shit.",
    formJobTitle: "Software Engineer",
    formPersonalJobTitle: "Software Engineer",
    formSchoolEndDates: ["Fri Jan 24 2025 00:00:00 GMT-0800 (Pacific Standard Time)"],
    formSchoolStartDates: ["Tue Jan 21 2025 00:00:00 GMT-0800 (Pacific Standard Time)"],
    formSchools: ['University of California, Irvine'],
    interviewMode: "Mixed Interview",
    jobLink: "",
    resumeFile: null,
    resumeName: "Ex. John_Doe.pdf"
  }
    */

  useEffect(() => {
    if (currentHTML === 7) {
      let API_LINK = API_URL + "init_interview_session";
      let body = JSON.stringify({
        mode: initializeData.interviewMode,
        custom_job_str: {
          formJobTitle: initializeData.formJobTitle,
          formEmployerName: initializeData.formEmployerName,
          formEmployerLocation: initializeData.formEmployerLocation,
          formJobDescription: initializeData.formJobDescription,
          formEmployerDescription: initializeData.formEmployerDescription,
        },
        interviewee_records: {
          formFullName: initializeData.formFullName,
          formPersonalJobTitle: initializeData.formPersonalJobTitle,
          formDegrees: initializeData.formDegrees,
          formSchools: initializeData.formSchools,
          formSchoolStartDates: initializeData.formSchoolStartDates,
          formSchoolEndDates: initializeData.formSchoolEndDates,

          formExperienceJobTitles: initializeData.formExperienceJobTitles,
          formExperienceStartDates: initializeData.formExperienceStartDates,
          formExperienceEndDates: initializeData.formExperienceEndDates,
          formExperienceDescriptions: initializeData.formExperienceDescriptions,
          formExperienceCompanies: initializeData.formExperienceCompanies,
          formExperienceLocations: initializeData.formExperienceLocations,
        },
        website_url: initializeData.jobLink,
        session_key: "test",
        interviewee_resume: initializeData.resumeFile,
      });
      fetch(API_LINK, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: body,
      })
        .then((response) => response.json())
        .then((data) => setCurrentQuestion(JSON.parse(data.body)["summary"]));
    }
  }, [currentHTML]);

  /*
  useEffect(() => {
    if (currentHTML === 7) {
      let API_LINK = 'https://6f74-128-195-97-184.ngrok-free.app/get_interview_question'
      let body = JSON.stringify(
        {
          user_input: 'I am the best person in the world',
          session_key: "test"
      }
    ) 
      fetch(API_LINK, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: body,
      }).then(response => response.json()).then(data => console.log(data))
    }
  }, [currentHTML])

  */

  useEffect(() => {
    if (recentVideoRecording !== "") {
      try {
        let API_LINK = API_URL + "get_interview_question";
        let body = JSON.stringify({
          user_input: recentVideoRecording,
          session_key: "test",
        });
        fetch(API_LINK, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
            "ngrok-skip-browser-warning": "true",
          },
          body: body,
        })
          .then((response) => response.json())
          .then((data) => {
            console.log(data);
            setCurrentQuestion(JSON.parse(data.body)["next_question"]);
          });
      } catch (error) {
        console.log(error);
      }
    }
  }, [recentVideoRecording]);

  /*
  useEffect(() => {
    console.log('route fired')
    let API_LINK = 'https://0577-128-195-97-184.ngrok-free.app/init_interview_session'
    let body = JSON.stringify(
      {
        mode: fakeData.interviewMode,
        custom_job_str: {
          formJobTitle: fakeData.formJobTitle,
          formEmployerName: fakeData.formEmployerName,
          formEmployerLocation: fakeData.formEmployerLocation,
          formJobDescription: fakeData.formJobDescription,
          formEmployerDescription: fakeData.formEmployerDescription,
        },
        website_url: fakeData.jobLink,
        session_key: "test"
    }
  ) 
    console.log(body)
    fetch(API_LINK, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'ngrok-skip-browser-warning': 'true'
      },
      body: body,
    }).then(response => response.json()).then(data => console.log(JSON.parse(data.body)['summary']))
  }, [])
  */

  return <div className="overflow-x-hidden">{HTML_PARTS[currentHTML]}</div>;
}
