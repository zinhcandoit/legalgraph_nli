/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        chatBg: '#212121',
        sidebarBg: '#171717',
        cardBg: '#2f2f2f',
        hoverBg: '#383838',
        inputBg: '#2f2f2f',
        accentGreen: '#10a37f',
        accentGreenHover: '#1a7f64',
        accentGold: '#f59e0b',
        borderDark: '#424242',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      }
    },
  },
  plugins: [],
}