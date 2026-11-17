import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration optimized for both Browser UI Preview and Tauri v2 Desktop
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: false,
  },
  envPrefix: ['VITE_', 'TAURI_'],
});
