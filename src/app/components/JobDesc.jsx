import { useState } from "react";

import JobForm from "./JobForm";
import { TextField } from "@mui/material";

export default function JobDesc({
  currentHTML,
  setCurrentHTML,
  jobLink,
  setJobLink,
  ...props
}) {
  const [formOpened, setFormOpened] = useState(false);

  return (
    <div className="w-screen min-h-screen flex flex-col py-10 px-6 sm:px-16 right-gradient">
      <nav className="flex">
        <button
          onClick={() => setCurrentHTML(1)}
          className="text-lg sm:text-2xl hover:underline"
        >
          ⬅️ back
        </button>
      </nav>

      <h1 className="text-2xl sm:text-4xl my-8 sm:my-16">
        first, let's talk about them!
      </h1>

      <div className="text-lg sm:text-xl px-4 sm:px-40 flex flex-col gap-y-4">
        <h1>enter your job post link:</h1>
        <TextField
          type="url"
          value={jobLink}
          onChange={(e) => setJobLink(e.target.value)}
          placeholder="ex. https://www.linkedin.com/jobs/view/1234"
          className="w-full"
        />

        <h1 className="text-lg sm:text-xl text-center my-6 sm:my-10 underline">
          or
        </h1>

        <h1 className="text-lg sm:text-xl">describe the job:</h1>

        <div
          onClick={() => setFormOpened(!formOpened)}
          className={`flex items-center justify-center gap-x-2 sm:gap-x-6 cursor-pointer hover:bg-white p-2 sm:p-4 rounded-md transition-all ${
            formOpened ? "bg-white" : ""
          }`}
        >
          <hr className="w-1/5 sm:w-2/5 border-black" />
          <h2 className="text-base sm:text-lg font-semibold tracking-wide">
            {formOpened ? "Close the Form" : "Open the Form"}
          </h2>

          <hr className="w-1/5 sm:w-2/6 border-black" />
          <span className="text-lg sm:text-xl">{formOpened ? "⬆️" : "⬇️"}</span>
        </div>

        {formOpened ? <JobForm {...props} /> : null}
      </div>

      <button
        onClick={() => setCurrentHTML(3)}
        className="hover:bg-blue-950 text-center mx-auto my-6 sm:my-8 px-4 sm:px-6 py-2 sm:py-3 bg-blue-800 text-white rounded-3xl hover:bg-black transition-all"
      >
        next step!
      </button>
    </div>
  );
}
