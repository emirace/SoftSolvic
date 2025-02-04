import { useState } from "react";

import { TextField } from "@mui/material";
import EducationSection from "./EducationSection";
import ExperienceSection from "./ExperienceSection";

export default function PersonalForm({
  formFullName,
  setFormFullName,
  formPersonalJobTitle,
  setFormPersonalJobTitle,
  formDegrees,
  setFormDegrees,
  formSchools,
  setFormSchools,
  formSchoolStartDates,
  setFormSchoolStartDates,
  formSchoolEndDates,
  setFormSchoolEndDates,
  formExperienceJobTitles,
  setFormExperienceJobTitles,
  formExperienceStartDates,
  setFormExperienceStartDates,
  formExperienceEndDates,
  setFormExperienceEndDates,
  formExperienceDescriptions,
  setFormExperienceDescriptions,
  formExperienceCompanies,
  setFormExperienceCompanies,
  formExperienceLocations,
  setFormExperienceLocations,
}) {
  const [numberOfEducation, setNumberOfEducation] = useState(1);
  const [numberOfJobs, setNumberOfJobs] = useState(1);

  return (
    <div className="flex flex-col mx-4 sm:mx-20 p-4 sm:p-6 rounded-md bg-white">
      {/* Full Name & Job Title */}
      <div className="flex flex-col sm:flex-row justify-between gap-4 sm:gap-x-6">
        <div className="flex flex-col gap-y-2 w-full sm:min-w-[500px]">
          <h1 className="my-2">
            Full Name <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formFullName}
            onChange={(e) => setFormFullName(e.target.value)}
            fullWidth
            placeholder="ex. John Doe"
          />
        </div>

        <div className="flex flex-col gap-y-2 w-full sm:min-w-[350px]">
          <h1 className="my-2">
            Job Title <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formPersonalJobTitle}
            onChange={(e) => setFormPersonalJobTitle(e.target.value)}
            fullWidth
            placeholder="ex. Software Engineer"
          />
        </div>
      </div>

      {/* Education Section */}
      <h1 className="mt-8 text-xl sm:text-2xl mb-4 font-semibold text-blue-800">
        Education
      </h1>
      {[...Array(numberOfEducation)].map((_, index) => (
        <EducationSection
          key={index}
          index={index}
          formDegrees={formDegrees}
          setFormDegrees={setFormDegrees}
          formSchools={formSchools}
          setFormSchools={setFormSchools}
          formSchoolStartDates={formSchoolStartDates}
          setFormSchoolStartDates={setFormSchoolStartDates}
          formSchoolEndDates={formSchoolEndDates}
          setFormSchoolEndDates={setFormSchoolEndDates}
        />
      ))}

      <div className="mt-4 flex justify-end">
        <button
          onClick={() => setNumberOfEducation(numberOfEducation + 1)}
          className="flex items-center gap-2 bg-blue-800 hover:bg-blue-950 transition-all text-white py-1 px-3 rounded-lg text-sm"
        >
          Add
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-5 h-5"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 4.5v15m7.5-7.5h-15"
            />
          </svg>
        </button>
      </div>

      {/* Experience Section */}
      <h1 className="mt-6 text-xl sm:text-2xl mb-4 font-semibold text-blue-800">
        Experience
      </h1>
      {[...Array(numberOfJobs)].map((_, index) => (
        <ExperienceSection
          key={index}
          index={index}
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
        />
      ))}

      <div className="mt-4 flex justify-end">
        <button
          onClick={() => setNumberOfJobs(numberOfJobs + 1)}
          className="flex items-center gap-2 bg-blue-800 hover:bg-blue-950 transition-all text-white py-1 px-3 rounded-lg text-sm"
        >
          Add
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-5 h-5"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 4.5v15m7.5-7.5h-15"
            />
          </svg>
        </button>
      </div>

      {/* Skills Section */}
      <h1 className="mt-6 text-xl sm:text-2xl mb-4 font-semibold text-blue-800">
        Skills
      </h1>
    </div>
  );
}
