# Take-a-Hike

Take-a-Hike maps stock-price behavior to real terrain patterns across Europe.

Have you ever wondered what stocks come to life would look like in a healthy way? Have you ever wanted to retrace the steps of your portfolio? Now you can! With Take-a-Hike you can pick your favourite stock and it will automatically calculate several paths in the world that have similar relief as the stock of your choice. Take-a-Hike can make you take a hike and explore the world!

Pipeline:
1. User enters a ticker and selects a time window (`1d`, `1w`, `1mo`).
2. Backend fetches Yahoo Finance close prices for that window.
3. Backend runs TERCOM-like terrain matching on EUDEM slope data.
4. Backend sends the top 10 candidate coordinates to Gemini with prompt:
	 `Here are 10 coordinates of places. Pick the one that fits the vibe of [COMPANY] best.`
5. Backend returns the Gemini-selected candidate and route.
6. Frontend plots the stock series, top candidates, their comparison with the stock, and selected route on OpenStreetMap.

## Architecture

- `frontend/`: Streamlit UI, no direct finance or terrain logic.
- `backend/`: FastAPI service with Yahoo data fetch, terrain estimation, Gemini call.
- `nginx/`: Reverse proxy.

Reverse proxy routes:
- `/` -> frontend (`streamlit`)
- `/api/` -> backend API (`FastAPI`)
- `/health` -> backend health endpoint

Only Nginx exposes a host port. Backend and frontend are internal to the Docker network.

## Repository Layout

- `backend/service.py`: API entrypoint (`POST /api/analyze`, `GET /health`)
- `backend/terrain_estimator.py`: terrain matching library + Typer CLI for local tests
- `backend/gemini_query.py`: Gemini prompt/query and response parsing
- `frontend/app.py`: Streamlit app that calls backend and renders chart/map
- `docker-compose.yaml`: multi-service orchestration
- `nginx/default.conf`: reverse proxy config

## Environment Variables

Create/update `backend/.env` with:

- `GEMINI_API_KEY=...` for Gemini selection
- `TERRAIN_MAP_PATH` is set by compose to `/data/eudem_slop_3035_europe.tif`

Notes:
- Yahoo Finance via `yfinance` does not require an API key.
- The EUDEM GeoTIFF is mounted from `backend/eud_cp_slop/` into `/data` in the backend container.

## Run With Docker Compose

From repository root:

```bash
docker compose up --build
```

Then open:

- App UI: `http://localhost/`
- Health: `http://localhost/health`

Example API call through Nginx:

```bash
curl -X POST http://localhost/api/analyze \
	-H 'Content-Type: application/json' \
	-d '{
		"symbol": "AAPL",
		"window": "1mo",
		"company": "Apple",
		"spacing_m": 25,
		"headings": 12,
		"random_samples": 300,
		"top_k": 20,
		"refine_iters": 2,
		"refine_step_px": 120
	}'
```

## Local (Non-Docker) Development

Install deps:

```bash
python -m pip install -r backend/requirements.txt -r frontend/requirements.txt
```

Run backend:

```bash
cd backend
uvicorn service:app --host 0.0.0.0 --port 8000
```

Run frontend:

```bash
cd frontend
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

## Terrain Estimator CLI (Quick Testing)

`backend/terrain_estimator.py` includes a Typer CLI for local estimation smoke tests.

Example:

```bash
cd backend
python terrain_estimator.py --symbol AAPL --window 1d --top-n 3 --samples 20 --headings 4
```

JSON output mode:

```bash
python terrain_estimator.py --symbol AAPL --window 1mo --json
```

## Notes and Limitations

- Terrain matching is stochastic; results can vary with search parameters.
- Higher `random_samples` and `headings` increase quality but cost more CPU time.
- If Gemini key is missing or fails, backend falls back to candidate rank 1 with an explanatory reason.
