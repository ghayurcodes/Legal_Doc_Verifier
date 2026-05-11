import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import About from './pages/About'
import HowToUse from './pages/HowToUse'
import Verify from './pages/Verify'

export default function App() {
  useEffect(() => {
    const handleMouseMove = (e) => {
      document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`)
      document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`)
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  return (
    <div className="app-shell glow-wrapper">
      <Navbar />
      <main className="page-content">
        <Routes>
          <Route path="/"          element={<About />} />
          <Route path="/how-to-use" element={<HowToUse />} />
          <Route path="/verify"    element={<Verify />} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}
