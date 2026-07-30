# FPL Assistant — Web (Phase 5.2)

React + Vite + TypeScript + Tailwind frontend over the FastAPI backend in `../api/`. Implements
the BUILD_PLAN 5.2 sitemap: Dashboard, My Team, Captaincy, Transfers, Chips, Fixtures, plus
honest "coming soon" placeholders for Player Search / Model Performance / Settings, whose backend
endpoints don't exist yet.

## Development

```bash
npm install
npm run dev
```

Requires the API running separately (`uv run uvicorn api.main:app --reload` from the repo root)
on `http://localhost:8000` by default — override with `VITE_API_BASE_URL` (see `.env.example`).

```bash
npm run build   # tsc -b && vite build
npm run lint    # oxlint
```
