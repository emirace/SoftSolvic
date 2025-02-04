import { TextField } from "@mui/material";
import { LocalizationProvider } from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import { DatePicker } from "@mui/x-date-pickers";

export default function EducationSection({
  index,
  formDegrees,
  setFormDegrees,
  formSchools,
  setFormSchools,
  formSchoolStartDates,
  setFormSchoolStartDates,
  formSchoolEndDates,
  setFormSchoolEndDates,
}) {
  function handleFormDegreeChange(event, index) {
    const newFormDegrees = [...formDegrees];
    newFormDegrees[index] = event.target.value;
    setFormDegrees(newFormDegrees);
  }

  function handleFormSchoolChange(event, index) {
    const newFormSchools = [...formSchools];
    newFormSchools[index] = event.target.value;
    setFormSchools(newFormSchools);
  }

  function handleFormSchoolStartDateChange(event, index) {
    const newFormSchoolStartDates = [...formSchoolStartDates];
    newFormSchoolStartDates[index] = event.$d;
    setFormSchoolStartDates(newFormSchoolStartDates);
  }

  function handleFormSchoolEndDateChange(event, index) {
    const newFormSchoolEndDates = [...formSchoolEndDates];
    newFormSchoolEndDates[index] = event.$d;
    setFormSchoolEndDates(newFormSchoolEndDates);
  }

  return (
    <>
      <div className="flex flex-col gap-y-2 mb-2 w-full sm:min-w-[350px]">
        <h1 className="my-2">
          Degree <span className="text-red-500">*</span>
        </h1>
        <TextField
          value={formDegrees[index]}
          onChange={(e) => handleFormDegreeChange(e, index)}
          fullWidth
          placeholder="ex. Software Engineer"
        />
      </div>

      <div className="flex flex-col sm:flex-row sm:justify-between sm:gap-x-6 gap-y-4 sm:items-center">
        <div className="flex flex-col gap-y-2 w-full sm:min-w-[500px]">
          <h1 className="my-2">
            School Name <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formSchools[index]}
            onChange={(e) => handleFormSchoolChange(e, index)}
            fullWidth
            placeholder="ex. John Doe"
          />
        </div>

        <div className="w-full sm:w-auto">
          <h1 className="my-2">
            Start Date <span className="text-red-500">*</span>
          </h1>
          <LocalizationProvider dateAdapter={AdapterDayjs}>
            <DatePicker
              onChange={(e) => handleFormSchoolStartDateChange(e, index)}
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
              onChange={(e) => handleFormSchoolEndDateChange(e, index)}
              className="text-gray-600 w-full"
            />
          </LocalizationProvider>
        </div>
      </div>

      <hr className="w-full mt-8 mb-4 border-gray-600" />
    </>
  );
}
