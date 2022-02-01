const defaultTheme = require('tailwindcss/defaultTheme')

module.exports = {
  mode: 'jit',
  content: ['./templates/**/*.html'],
  theme: {
    extend: {},
    fontFamily: {
      'sans': ['Dongle']
    },
    fontSize: {
      'xs': '1rem',
      'sm': '1.25rem',
      'tiny': '1.5rem',
      'base': '2rem',
      'lg': '2.5rem',
      'xl': '3rem',
      '2xl': '3.5rem',
      '3xl': '4.5rem',
      '4xl': '5rem',
      '5xl': '6rem',
      '6xl': '8rem',
      '7xl': '10rem',
    },
  },
  plugins: [
  ],
}
