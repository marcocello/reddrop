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
