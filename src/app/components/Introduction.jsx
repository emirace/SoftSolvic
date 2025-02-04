import { useEffect } from "react"
import { TextEffectOne } from "react-text-animate";

export default function Introduction({ currentHTML, setCurrentHTML}) {

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentHTML(currentHTML + 1)
            clearTimeout(interval)
        }, 7000)
    }, [])

    return (
        <div className="w-screen h-screen flex flex-col items-center justify-center gap-y-16 left-gradient">
            <h1 className="text-5xl animate-fadeOut">
                <TextEffectOne text='hi there 👋' animateOnce staggerDuration={.05} />
            </h1>
            <h1 className="text-5xl animate-fadeOut">
                <TextEffectOne text="it's nice to meet you!" lineHeight={1.5} initialDelay={1} animateOnce staggerDuration={.075} />
            </h1>
        </div>
    )
}