import { useState } from "react";

import PersonalForm from "./PersonalForm";

export default function PersonalDesc({
  currentHTML,
  setCurrentHTML,
  resumeName,
  setResume,
  setResumeFile,
  ...props
}) {
  const [formOpened, setFormOpened] = useState(false);

  function handleFileChange(event) {
    console.log("call changed");
    const selectedFile = event.target.files[0];

    const reader = new FileReader();
    reader.onload = (event) => {
      const arrayBuffer = event.target.result;
      const byteArray = new Uint8Array(arrayBuffer);
      const bytes = Array.from(byteArray);
      setResumeFile(bytes);
      console.log(bytes);
    };
    reader.readAsArrayBuffer(selectedFile);
    setResume(selectedFile.name);
  }

  return (
    <div className="w-screen min-h-screen flex flex-col py-10 px-6 sm:px-16 right-gradient">
      <nav className="flex">
        <button
          onClick={() => setCurrentHTML(2)}
          className="text-2xl hover:underline"
        >
          ⬅️ back
        </button>
      </nav>

      <h1 className="text-4xl my-12 text-center sm:text-left">
        Next, let's talk about you!
      </h1>

      <div className="text-xl w-full max-w-4xl mx-auto flex flex-col gap-y-6">
        {/* Resume Upload Section */}
        <div className="flex flex-col sm:flex-row justify-between bg-white px-6 sm:px-10 py-4 rounded-lg items-center">
          <h1>Import Resume</h1>
          <div className="flex flex-col sm:flex-row gap-y-2 sm:gap-x-4 items-center">
            <p className="text-sm text-gray-400 truncate max-w-[200px]">
              {resumeName || "No file selected"}
            </p>
            <label
              htmlFor="resume-upload"
              className="cursor-pointer flex items-center gap-x-2 bg-blue-800 hover:bg-blue-950 transition-all text-white p-2 rounded-lg"
            >
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
                  d="m18.375 12.739-7.693 7.693a4.5 4.5 0 0 1-6.364-6.364l10.94-10.94A3 3 0 1 1 19.5 7.372L8.552 18.32m.009-.01-.01.01m5.699-9.941-7.81 7.81a1.5 1.5 0 0 0 2.112 2.13"
                />
              </svg>
              Attach File
            </label>
            <input
              onChange={handleFileChange}
              type="file"
              accept="application/pdf"
              id="resume-upload"
              className="hidden"
            />
          </div>
        </div>

        <h1 className="text-xl text-center underline">or</h1>

        {/* Personal Description Section */}
        <h1 className="text-xl">Describe Yourself:</h1>

        <div
          onClick={() => setFormOpened(!formOpened)}
          className="bg-white flex justify-between items-center w-full max-w-xs sm:max-w-md mx-auto cursor-pointer p-3 sm:p-4 rounded-md transition-all shadow-md"
        >
          <hr className="w-1/6 sm:w-1/4 border-black" />
          <h2 className="text-sm sm:text-lg font-semibold tracking-wide whitespace-nowrap">
            {formOpened ? "Close the Form" : "Open the Form"}
          </h2>
          <hr className="w-1/6 sm:w-1/4 border-black" />
          <span className="text-lg">{formOpened ? "⬆️" : "⬇️"}</span>
        </div>

        {formOpened && <PersonalForm {...props} />}
      </div>

      <button
        onClick={() => setCurrentHTML(4)}
        className="hover:bg-blue-950 text-center mx-auto my-8 mt-12 px-6 py-3 bg-blue-800 text-white rounded-3xl transition-all"
      >
        Next Step!
      </button>
    </div>
  );
}
