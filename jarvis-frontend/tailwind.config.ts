import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "#09101d",
        accent: "#8dd3ff",
        ink: "#e8f0ff"
      },
      boxShadow: {
        glow: "0 0 40px rgba(141, 211, 255, 0.18)"
      }
    }
  },
  plugins: []
} satisfies Config;
