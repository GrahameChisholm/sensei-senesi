import { useState } from "react";
import { PlayerStats } from "./pages/PlayerStats";
import { TeamSelection } from "./pages/TeamSelection";
import { FixturesPage } from "./pages/FixturesPage";
import { Differentials } from "./pages/Differentials";
import { MiniLeague } from "./pages/MiniLeague";

type Tab = "team" | "fixtures" | "stats" | "differentials" | "mini-league";

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
        <button
          className={tab === "differentials" ? "active" : ""}
          onClick={() => setTab("differentials")}
        >
          Differentials
        </button>
        <button
          className={tab === "mini-league" ? "active" : ""}
          onClick={() => setTab("mini-league")}
        >
          Mini League
        </button>
      </nav>
      {tab === "team" ? (
        <TeamSelection />
      ) : tab === "fixtures" ? (
        <FixturesPage />
      ) : tab === "stats" ? (
        <PlayerStats />
      ) : tab === "differentials" ? (
        <Differentials />
      ) : (
        <MiniLeague />
      )}
    </div>
  );
}
