import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        "tm-bg": "#0a0a0f",
        "tm-card": "#13131a",
        "tm-border": "#1e1e2e",
        "tm-accent": "#6c5ce7",
        "tm-green": "#00d4aa",
        "tm-red": "#ff6b6b",
        "tm-yellow": "#ffd93d",
        "tm-blue": "#4dabf7",
        "tm-purple": "#a855f7",
        "tm-text": "#e4e4e7",
        "tm-muted": "#71717a",
      },
    },
  },
  plugins: [],
};
export default config;
