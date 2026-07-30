import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Nav } from './components/Nav'
import { Dashboard } from './pages/Dashboard'
import { MyTeam } from './pages/MyTeam'
import { Captaincy } from './pages/Captaincy'
import { Transfers } from './pages/Transfers'
import { Chips } from './pages/Chips'
import { Fixtures } from './pages/Fixtures'
import { ComingSoon } from './pages/ComingSoon'

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/team" element={<MyTeam />} />
        <Route path="/captaincy" element={<Captaincy />} />
        <Route path="/transfers" element={<Transfers />} />
        <Route path="/chips" element={<Chips />} />
        <Route path="/fixtures" element={<Fixtures />} />
        <Route
          path="/players"
          element={
            <ComingSoon
              title="Player Search"
              reason="No player-search/comparison endpoint exists yet -- the API currently only serves full-pool rankings (Captaincy) rather than single-player lookup."
            />
          }
        />
        <Route
          path="/performance"
          element={
            <ComingSoon
              title="Model Performance"
              reason="Surfaces Phase 3's backtest metrics (MAE, calibration, captaincy hit-rate) once they're exposed via the API -- that wiring hasn't been built yet."
            />
          }
        />
        <Route
          path="/settings"
          element={
            <ComingSoon
              title="Settings"
              reason="FPL team ID, mini-league ID, and planning-horizon-default settings need persistent storage, which doesn't exist yet -- the demo state is currently fixed."
            />
          }
        />
      </Routes>
    </BrowserRouter>
  )
}

export default App
