import { TextEffectOne } from "react-text-animate";
import { useEffect } from "react";
import { Slide } from "@mui/material";

export default function GetStarted({ currentHTML, setCurrentHTML }) {
  return (
    <div className="w-full min-h-screen flex flex-col items-center justify-center gap-y-8 px-4 text-center left-gradient">
      <h1 className="text-3xl md:text-5xl">
        <TextEffectOne text="Welcome to" animateOnce staggerDuration={0.05} />
      </h1>
      <h1 className="text-6xl md:text-9xl tracking-widest">
        <TextEffectOne
          text="SoftSolvic"
          lineHeight={1.2}
          initialDelay={1}
          animateOnce
          staggerDuration={0.075}
        />
      </h1>
      <Slide direction="up" in={true} mountOnEnter timeout={1000}>
        <button
          className="text-lg md:text-xl mt-8 md:mt-16 hover:underline"
          onClick={() => {
            setCurrentHTML(2);
          }}
        >
          Get Started <span>👉</span>
        </button>
      </Slide>
    </div>
  );
}
