export default function InterviewTypeCard({ Icon, Title, onMouseEnter, onMouseLeave}) {
    return (
        <button onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave} className="flex flex-col items-center gap-y-12 bg-white border-blue-800 stroke-blue-800 text-blue-800 border-2 rounded-lg py-10 px-16">
            {Icon}
            {Title}
        </button>
    )
}