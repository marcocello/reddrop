# Reddrop Architecture

## Components
- `./reddrop`: single CLI entrypoint.
- `backend/cli/reddrop.py`: parser, completion, and command dispatch wiring.
- `backend/cli/commands/add.py`: `add` command handler.
- `backend/cli/commands/search.py`: `search` command handler.
- `backend/cli/commands/reply.py`: `reply` command handler.
- `backend/cli/commands/send.py`: `send` command handler.
- `backend/api/main.py`: FastAPI layer exposing command-equivalent operations.
- `backend/api/job_runtime.py`: in-memory job process manager for start/stop/restart status.
- `backend/services/`: discovery, persistence, LLM, and persona access.
- `backend/services/settings_store.py`: load/save `.env` runtime settings for Reddit/OpenRouter integration.
- `backend/config/log.py`: shared logging configuration and `setup_logging`.
- `backend/models.py`: job/search data contracts.
- `frontend/`: React + Vite dashboard derived from `satnaing/shadcn-admin`, trimmed to jobs/personas/threads/settings views.
- `backend/Dockerfile`: backend image definition for FastAPI/CLI runtime.
- `deployment/docker-compose.yml`: local backend container wiring.

## Command Boundaries
- `add`: create or update one job configuration identified by `name`.
- `search`: execute discovery for all jobs or one named job and persist one search artifact file per job.
- `reply`: read a search artifact and write generated drafts into the same artifact conversations.
- `send`: read a search artifact, post unsent drafted replies, and persist send status updates in the same artifact.
- `reply` and `send` return compact CLI summaries and do not print full artifact payloads.

## Data Contracts
- Job config file: `.reddrop/jobs_config.json`.
- Search artifacts: `.reddrop/runs/search_<job_id_prefix8>.json` (one file per job).
- Job `id` values are UUIDs generated automatically.
- Job entries store `id`, `name`, `topic`, `time_filter`, `subreddit_limit`, `threads_limit`, `replies_per_iteration`, `max_runtime_minutes`, and `personas`; `name` is a no-space identifier (`[A-Za-z0-9_-]+`).
- Search artifact payload includes `id` (job id), `created_at`, `updated_at`, and merged `conversations`.
- Conversation rows include `reply` and `user_has_commented`, so search/reply/send state is unified in one file per job.

## Integration Rules
- Discovery reuses `backend.services.reddit_service.RedditService.discover_relevant_threads`.
- `search` first writes an initialized artifact (empty conversations) and then updates the same job file on completion.
- Conversation merge behavior is keyed by (`thread_id`, `subreddit`) with incoming matches replacing previous entries.
- `search` accepts an optional job `name` positional argument; without it, all jobs are executed.
- Discovery tuning flags are accepted by `add` and persisted in job config.
- `search` does not accept discovery tuning flags and always uses job-config values.
- `reply` accepts a search file argument and `--replies` limit (default `3`).
- `reply` requires `--personas <name>` and resolves persona data from `personas.json` or `.reddrop/personas.json`.
- `reply` uses `ReplyGenerationService` for draft text and writes drafts into `conversation.reply` in `search_*.json`.
- `reply` selection excludes conversations that already have a non-empty `reply`, applies a minimum similarity threshold (`semantic_similarity >= 0.35`), and reranks remaining entries by `semantic_similarity` descending before taking top `--replies`.
- `search` persistence is handled by `SearchStorage` (`backend/services/search_storage.py`).
- `SearchStorage` provides file locking (`flock`) + atomic replace writes for all search/reply/send updates.
- `ReplyGenerationService` extracts tone/style from thread title/body during `reply` and applies it to final reply generation.
- Discovery (`search`) does not perform tone/style extraction.
- `send` uses `RedditService.post_comment` and updates `user_has_commented` in the same `search_*.json` artifact.
- `reply` emits `search_file`, `reply_file`, `generated`, `total_with_replies`.
- `send` emits `search_file`, `reply_file`, `sent`, `already_commented`, `candidates`, `total`.
- FastAPI exposes `/health`, `/jobs/add`, `/search`, `/reply`, and `/send`, using the same command handlers/dependencies as CLI.
- FastAPI also exposes API-first workflow endpoints for frontend-only usage:
  - `POST /jobs`, `DELETE /jobs/{job}`, `POST /jobs/{job}/start`, `POST /jobs/{job}/stop`
  - `PUT /jobs/{job}` exists but returns `405` because jobs are immutable after creation
  - `GET /personas`, `PUT /personas` for persona catalog management
  - `GET /settings`, `PUT /settings` for runtime Reddit/OpenRouter settings management
  - `GET /threads` with JSON-artifact filters (`job_id`, `job_name`, `subreddit`, `min_similarity`, `only_open`, `has_reply`, paging)
  - `POST /threads/{job}/{subreddit}/{thread_id}/reply` and `POST /threads/{job}/{subreddit}/{thread_id}/send` for per-thread actions
- FastAPI also exposes job lifecycle and artifact browsing endpoints (`/jobs`, `/jobs/status`, `/jobs/{name}/start|stop|restart`, `/artifacts/search*`, `/artifacts/reply*`) for frontend usage.
- Job start/restart actions spawn CLI `search` subprocesses; stop issues process termination and status is tracked in-memory.
- CLI completion scripts (`bash`, `zsh`) resolve search-name candidates from configured job names and must work for both `reddrop` and `./reddrop`.
- Discovery progress is emitted via standard info logs from CLI/services.
- Discovery logs include output summaries for analysis intent/topics/keywords, discovered subreddits, generated queries, and candidate thread counts.
- `backend/services/llm_service.py` uses OpenRouter (`OPENROUTER_*`) and falls back to deterministic demo responses when missing/unavailable.
- Modules that log should use `logger = setup_logging(__name__)` for a shared formatter/level policy.
- Frontend consumes API only; no business logic is implemented in UI components.
- Jobs UI supports create + lifecycle controls (activate/deactivate/start/stop/delete); existing jobs are read-only in details view.
- Frontend threads/reply views are driven by `GET /threads` plus thread-level reply/send actions, rather than direct artifact file selection.
- Table-based frontend views use shadcn-style toolbar + bordered table sections; create actions use modal dialogs while row details/editors open in right-side sheets.
- Destructive frontend actions (job/persona deletion) use explicit confirmation dialogs before API calls are executed.
- Threads UI uses shadcn data-table patterns (toolbar filters/search/pagination) and exposes row actions via a fixed right-most three-dots menu column.
- Sortable table headers trigger direct click-based asc/desc toggling (no header sort context menu), and table columns support drag-based reordering via TanStack `columnOrder` state.
- Threads UI does not expose a free-form persona input; reply actions resolve persona from the selected job configuration.
- Threads sidebar metadata surfaces similarity/score/comments/created with badge status, fixed-height body content, and reply input actions below the input.
- Frontend includes a dedicated Personas page using the same table + modal-create + sheet-edit interaction model as Jobs, persisted via `/personas`.
- Frontend includes a dedicated Settings page for Reddit/OpenRouter fields, persisted via `/settings`.
- Frontend does not expose a dedicated Replies page; reply drafting/sending is handled within Threads workflows.
- Jobs and Threads headers include a shared auto-refresh dropdown (`off`, `5s`, `30s`, `1m`) that controls polling intervals per page.
- Template sections not used by Reddrop are removed from active `src/` routes/features and parked under `frontend/_unused/`.
