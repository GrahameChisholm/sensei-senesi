import { useState } from "react";
import { PlayerStats } from "./pages/PlayerStats";
import { TeamSelection } from "./pages/TeamSelection";
import { FixturesPage } from "./pages/FixturesPage";

type Tab = "team" | "fixtures" | "stats";

export default function App() {
  const [tab, setTab] = useState<Tab>("team");

  return (
    <div>
      <nav className="app-tabs">
        <button className={tab === "team" ? "active" : ""} onClick={() => setTab("team")}>
          Team
        </button>
        <button className={tab === "fixtures" ? "active" : ""} onClick={() => setTab("fixtures")}>
          Fixtures
        </button>
        <button className={tab === "stats" ? "active" : ""} onClick={() => setTab("stats")}>
          Player Stats
        </button>
      </nav>
      {tab === "team" ? <TeamSelection /> : tab === "fixtures" ? <FixturesPage /> : <PlayerStats />}
    </div>
  );
}
