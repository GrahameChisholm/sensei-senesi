import { useState } from "react";
import { PlayerStats } from "./pages/PlayerStats";
import { TeamSelection } from "./pages/TeamSelection";

type Page = "team" | "stats";

export default function App() {
  const [page, setPage] = useState<Page>("team");

  return (
    <div>
      <nav className="app-nav">
        <button className={page === "team" ? "active" : ""} onClick={() => setPage("team")}>
          Team Selection
        </button>
        <button className={page === "stats" ? "active" : ""} onClick={() => setPage("stats")}>
          Player Stats
        </button>
      </nav>
      {page === "team" ? <TeamSelection /> : <PlayerStats />}
    </div>
  );
}
