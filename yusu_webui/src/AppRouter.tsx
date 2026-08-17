import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import { useSettingsStore } from '@/stores/settings'
import App from './App'
import LoginPage from '@/features/LoginPage'
import ThemeProvider from '@/components/ThemeProvider'
import '@/lib/extensions'

const AppContent = () => {
  const baseUrl = useSettingsStore.use.baseUrl()

  if (!baseUrl) {
    return (
      <Routes>
        <Route path="*" element={<LoginPage />} />
      </Routes>
    )
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/*" element={<App />} />
    </Routes>
  )
}

const AppRouter = () => {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
        <Toaster position="bottom-center" theme="system" closeButton richColors />
      </Router>
    </ThemeProvider>
  )
}

export default AppRouter