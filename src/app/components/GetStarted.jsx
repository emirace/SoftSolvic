import { TextEffectOne } from "react-text-animate";
import { useEffect } from "react";
import { Slide } from "@mui/material";

export default function GetStarted({ currentHTML, setCurrentHTML }) {
  return (
    <div className="w-screen h-screen flex flex-col items-center justify-center gap-y-16 left-gradient">
      <h1 className="text-5xl">
        <TextEffectOne text="welcome to" animateOnce staggerDuration={0.05} />
      </h1>
      <h1 className="text-9xl tracking-widest">
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
          className="text-xl mt-16 hover:underline"
          onClick={() => {
            setCurrentHTML(2);
          }}
        >
          get started <span>👉</span>
        </button>
      </Slide>
    </div>
  );
}
