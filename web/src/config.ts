// The gameweek the dashboard/chips pages default to. Real deployment would read this from the
// backend (the next unplayed gameweek) once Phase 6's weekly job exists; hardcoded for now since
// the demo state (api/demo_data.py) always represents the same fixed 5-gameweek horizon.
export const DEFAULT_GAMEWEEK = 1
