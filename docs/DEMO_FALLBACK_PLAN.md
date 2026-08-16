# Demo Fallback Plan
## Primary plan
Run `docker compose up --build` from a laptop that already has the images
built (build them the night before, not live).
## If Docker fails or is too slow live
1. Fall back to running the backend directly: `cd backend && uvicorn app.main:app --reload`
2. Run the frontend directly: `cd frontend && npm run dev`
3. Both should already be tested and working from Week 3 - this skips
Docker entirely and is faster to start.
## If the network/wifi fails entirely
Use the screen recording captured on Day 19 (see docs/screenshots/ and any
recorded demo video) and narrate over it instead of a live demo.
## Who owns this
Whoever's laptop is being used for the demo should have the backend and
frontend already running and tested *before* your review slot starts, not
started live.
