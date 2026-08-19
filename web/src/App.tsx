import { useState } from "react";
import { TeamSelection } from "./pages/TeamSelection";
import { FixturesPage } from "./pages/FixturesPage";

type Tab = "team" | "fixtures";

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
      </nav>
      {tab === "team" ? <TeamSelection /> : <FixturesPage />}
    </div>
  );
}
