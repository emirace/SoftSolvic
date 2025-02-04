/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      keyframes: {
        fadeOut: {
          "0%": { opacity: 1 },
          "100%": { opacity: 0 },
        },
        fadeIn: {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 },
        },
    },
    animation: {
      fadeOut: "fadeOut 1s ease-in-out 4s forwards",
      fadeOut8s: "fadeOut 1s ease-in-out 8s forwards",
      fadeOut28s: "fadeOut 1s ease-in-out 28s forwards",
    }
  },
},
  plugins: [],
};
