import { TextEffectOne } from "react-text-animate"
import { CircularProgress } from "@mui/material"
import { useEffect } from "react"
import { useState } from "react"

import Fade from "@mui/material/Fade"

export default function InformationLoad({ currentHTML, setCurrentHTML }) {

    const [loading, setLoading] = useState(false)

    useEffect(() => {
        
        const interval = setInterval(() => {
            setCurrentHTML(currentHTML + 1)
            clearTimeout(interval)
        }, 10000)

        const interval2 = setInterval(() => {
            setLoading(true)
            clearTimeout(interval2)
        }, 3000)

        const interval3 = setInterval(() => {
            setLoading(false)
            clearTimeout(interval3)
        }, 8000)

    }, [])

    return (
        <div className="text-blue-900 text-5xl w-screen h-screen flex flex-col items-center justify-center gap-y-16 left-gradient">
            <h1 className="text-5xl animate-fadeOut8s">
                <TextEffectOne text='all done!' animateOnce staggerDuration={.05} />
            </h1>
            <h1 className="text-5xl animate-fadeOut8s">
                <TextEffectOne text="thanks for filling that out" lineHeight={1.5} initialDelay={1} animateOnce staggerDuration={.075} />
            </h1>
            <Fade timeout={2000} in={loading}>
                <CircularProgress />
            </Fade>
        </div>
    )
}