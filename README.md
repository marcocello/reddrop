<div style="text-align: left;">
  <img src="logo.svg" alt="Reddrop" width="180"/>
</div>

Find high-signal Reddit threads for your topic, generate persona-aware replies, and ship faster.

Reddrop helps you:
- discover relevant threads for a topic
- draft replies that match your persona + thread context
- send replies and track what was posted
- run jobs repeatedly with runtime settings

**How To Run**

Backend 
```bash
docker compose -f deployment/docker-compose.yml up --build backend
```

Backend URLs:
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

Frontend

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Frontend URL:
- `http://127.0.0.1:5173`

**CLI (optional)**

```bash
./backend/reddrop --help
```

Main commands:
- `add`
- `search`
- `reply`
- `send`

**License**

MIT
