import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { DatePicker } from "@mui/x-date-pickers";
import { TextField } from "@mui/material";

export default function ExperienceSection({
  index,
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
  function handleFormExperienceJobTitleChange(e) {
    const updatedFormExperienceJobTitles = [...formExperienceJobTitles];
    updatedFormExperienceJobTitles[index] = e.target.value;
    setFormExperienceJobTitles(updatedFormExperienceJobTitles);
  }

  function handleFormExperienceStartDateChange(e) {
    const updatedFormExperienceStartDates = [...formExperienceStartDates];
    updatedFormExperienceStartDates[index] = e.$d;
    setFormExperienceStartDates(updatedFormExperienceStartDates);
  }

  function handleFormExperienceEndDateChange(e) {
    const updatedFormExperienceEndDates = [...formExperienceEndDates];
    updatedFormExperienceEndDates[index] = e.$d;
    setFormExperienceEndDates(updatedFormExperienceEndDates);
  }

  function handleFormExperienceDescriptionChange(e) {
    const updatedFormExperienceDescriptions = [...formExperienceDescriptions];
    updatedFormExperienceDescriptions[index] = e.target.value;
    setFormExperienceDescriptions(updatedFormExperienceDescriptions);
  }

  function handleFormExperienceCompanyChange(e) {
    const updatedFormExperienceCompanies = [...formExperienceCompanies];
    updatedFormExperienceCompanies[index] = e.target.value;
    setFormExperienceCompanies(updatedFormExperienceCompanies);
  }

  function handleFormExperienceLocationChange(e) {
    const updatedFormExperienceLocations = [...formExperienceLocations];
    updatedFormExperienceLocations[index] = e.target.value;
    setFormExperienceLocations(updatedFormExperienceLocations);
  }

  return (
    <>
      <div className="flex flex-col sm:flex-row sm:justify-between gap-x-6 items-center">
        <div className="flex flex-col gap-y-2 w-full sm:min-w-[500px]">
          <h1 className="my-2">
            Job Title <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formExperienceJobTitles[index]}
            onChange={handleFormExperienceJobTitleChange}
            fullWidth
            placeholder="ex. Job Title"
          />
        </div>

        <div className="w-full sm:w-auto">
          <h1 className="my-2">
            Start Date <span className="text-red-500">*</span>
          </h1>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <DatePicker
              onChange={handleFormExperienceStartDateChange}
              className="text-gray-600 w-full"
            />
          </LocalizationProvider>
        </div>

        <div className="w-full sm:w-auto">
          <h1 className="my-2">
            End Date <span className="text-red-500">*</span>
          </h1>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <DatePicker
              onChange={handleFormExperienceEndDateChange}
              className="text-gray-600 w-full"
            />
          </LocalizationProvider>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:justify-between gap-x-6 my-2">
        <div className="flex flex-col gap-y-2 w-full sm:min-w-[500px]">
          <h1 className="my-2">
            Employer <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formExperienceCompanies[index]}
            onChange={handleFormExperienceCompanyChange}
            fullWidth
            placeholder="ex. Company Name"
          />
        </div>

        <div className="flex flex-col gap-y-2 w-full sm:min-w-[370px]">
          <h1 className="my-2">
            Location <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formExperienceLocations[index]}
            onChange={handleFormExperienceLocationChange}
            fullWidth
            placeholder="ex. Location"
          />
        </div>
      </div>

      <h1 className="my-2 mt-6">
        Job Description <span className="text-red-500">*</span>
      </h1>
      <textarea
        value={formExperienceDescriptions[index]}
        onChange={handleFormExperienceDescriptionChange}
        className="border-2 h-[300px] resize-none mb-4 rounded-md p-3 px-4 text-sm w-full"
      />

      <hr className="w-full mt-8 mb-4 border-gray-600" />
    </>
  );
}
