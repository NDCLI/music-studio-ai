/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        studio: {
          bg: '#080B14',
          surface: '#0F1424',
          card: '#151C32',
          border: 'rgba(255, 255, 255, 0.09)',
          primary: '#4F46E5',
          cta: '#22C55E',
          lead: '#EC4899',
          backing: '#F59E0B',
          inst: '#10B981',
          reverb: '#06B6D4',
        },
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
        display: ['"Righteous"', '"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
