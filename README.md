RiteSignal
Find the high-signal conversations. Post comments that make people care.

Most teams underestimate comments.
People get interested when someone explains why a product matters in plain language, inside a real discussion.
That is why comments often outperform polished posts: they feel human, specific, and timely.

RiteSignal finds high-signal conversations across Reddit and LinkedIn, drafts context-aware replies, and helps you engage consistently.

RiteSignal helps you:
- find relevant Reddit threads and LinkedIn posts for your topic
- identify interesting conversations worth joining
- draft replies that match your persona and thread context
- post and track what was sent
- run repeatable jobs with runtime settings

**How To Run**

Docker (build + serve frontend and backend)
```bash
docker compose -f deployment/docker-compose.yml up --build
```

Frontend URL:
- local: `http://127.0.0.1:5173`
- LAN: `http://<SERVER_LAN_IP>:5173`

Backend URLs:
- API direct: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Frontend calls backend through `/api` via Nginx proxy

No Node install is required on the server.

Find your server LAN IP (macOS):

```bash
ipconfig getifaddr en0
```

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
