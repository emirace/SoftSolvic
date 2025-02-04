import { TextField } from "@mui/material";

export default function JobForm({
  formJobTitle,
  formEmployerName,
  formEmployerLocation,
  formJobDescription,
  formEmployerDescription,
  setFormJobTitle,
  setFormEmployerName,
  setFormEmployerLocation,
  setFormJobDescription,
  setFormEmployerDescription,
}) {
  return (
    <div className="flex flex-col mx-4 sm:mx-20 p-6 rounded-md bg-white max-w-lg sm:max-w-2xl lg:max-w-4xl mx-auto">
      <h1 className="my-2">
        Job Title <span className="text-red-500">*</span>
      </h1>
      <TextField
        value={formJobTitle}
        onChange={(e) => setFormJobTitle(e.target.value)}
        placeholder="ex. Software Engineer"
        className="w-full"
      />

      <div className="flex flex-col sm:flex-row justify-between gap-y-4 sm:gap-x-6 mt-3 mb-3">
        <div className="flex flex-col gap-y-2 w-full sm:w-1/2">
          <h1 className="my-2">
            Employer Name <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formEmployerName}
            onChange={(e) => setFormEmployerName(e.target.value)}
            placeholder="ex. John Doe"
            className="w-full"
          />
        </div>

        <div className="flex flex-col gap-y-2 w-full sm:w-1/2">
          <h1 className="my-2">
            Location <span className="text-red-500">*</span>
          </h1>
          <TextField
            value={formEmployerLocation}
            onChange={(e) => setFormEmployerLocation(e.target.value)}
            placeholder="ex. New York, USA"
            className="w-full"
          />
        </div>
      </div>

      <h1 className="my-2">
        Job Description <span className="text-red-500">*</span>
      </h1>
      <textarea
        value={formJobDescription}
        onChange={(e) => setFormJobDescription(e.target.value)}
        className="border-2 h-[200px] sm:h-[300px] resize-none rounded-md p-3 px-4 text-sm w-full"
      />

      <h1 className="my-2 mt-6">
        Employer Description <span className="text-red-500">*</span>
      </h1>
      <textarea
        value={formEmployerDescription}
        onChange={(e) => setFormEmployerDescription(e.target.value)}
        className="border-2 h-[200px] sm:h-[300px] resize-none mb-4 rounded-md p-3 px-4 text-sm w-full"
      />
    </div>
  );
}
