import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Nav } from './components/Nav'
import { DataFreshnessBanner } from './components/DataFreshnessBanner'
import { Dashboard } from './pages/Dashboard'
import { MyTeam } from './pages/MyTeam'
import { Captaincy } from './pages/Captaincy'
import { Transfers } from './pages/Transfers'
import { Chips } from './pages/Chips'
import { Fixtures } from './pages/Fixtures'
import { Settings } from './pages/Settings'
import { ModelPerformance } from './pages/ModelPerformance'
import { PlayerSearch } from './pages/PlayerSearch'

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <DataFreshnessBanner />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/team" element={<MyTeam />} />
        <Route path="/captaincy" element={<Captaincy />} />
        <Route path="/transfers" element={<Transfers />} />
        <Route path="/chips" element={<Chips />} />
        <Route path="/fixtures" element={<Fixtures />} />
        <Route path="/players" element={<PlayerSearch />} />
        <Route path="/performance" element={<ModelPerformance />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
