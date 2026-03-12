from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


PhaseCallback = Callable[[str], None]
Runner = Callable[[PhaseCallback], str | None]


@dataclass
class RuntimeLogEntry:
    at: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"at": self.at, "message": self.message}


@dataclass
class JobRunStatus:
    name: str
    active: bool = False
    status: str = "inactive"
    last_run_status: str = "never"
    successful_runs: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    last_error: str | None = None
    last_output: str | None = None
    logs: list[RuntimeLogEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "active": self.active,
            "status": self.status,
            "last_run_status": self.last_run_status,
            "successful_runs": self.successful_runs,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_error": self.last_error,
            "last_output": self.last_output,
            "logs": [item.to_dict() for item in self.logs],
        }


@dataclass
class JobRegistration:
    interval_minutes: int
    runner: Runner


class JobRuntimeManager:
    MAX_LOGS_PER_JOB = 200

    def __init__(self) -> None:
        self._lock = Lock()
        self._states: dict[str, JobRunStatus] = {}
        self._running: set[str] = set()
        self._stop_requested: set[str] = set()
        self._registrations: dict[str, JobRegistration] = {}
        self._scheduler = BackgroundScheduler(timezone="UTC", daemon=True)
        self._scheduler.start()

    def register(self, name: str, *, interval_minutes: int, runner: Runner) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            raise RuntimeError("Job name is required.")
        safe_interval = max(1, int(interval_minutes))
        with self._lock:
            self._registrations[normalized_name] = JobRegistration(interval_minutes=safe_interval, runner=runner)
            self._status_for_locked(normalized_name)
            self._append_log_locked(normalized_name, f"Scheduler registered (interval_minutes={safe_interval}).")
            if self._states[normalized_name].active:
                self._schedule_locked(normalized_name)

    def unregister(self, name: str) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            return
        with self._lock:
            self._registrations.pop(normalized_name, None)
            self._remove_schedule_locked(normalized_name)
            self._stop_requested.discard(normalized_name)
            state = self._status_for_locked(normalized_name)
            state.active = False
            if normalized_name in self._running:
                state.status = "stopping"
                self._append_log_locked(normalized_name, "Scheduler unregistered while run is active.")
            else:
                state.status = "inactive"
                state.finished_at = _now_iso()
                self._append_log_locked(normalized_name, "Scheduler unregistered.")

    def update_interval(self, name: str, interval_minutes: int) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            return
        with self._lock:
            registration = self._registrations.get(normalized_name)
            if registration is None:
                return
            registration.interval_minutes = max(1, int(interval_minutes))
            if self._states.get(normalized_name, JobRunStatus(name=normalized_name)).active:
                self._schedule_locked(normalized_name)

    def list_statuses(self, names: list[str]) -> list[dict[str, object]]:
        with self._lock:
            return [self._status_for_locked(name).to_dict() for name in names]

    def status(self, name: str) -> dict[str, object]:
        with self._lock:
            return self._status_for_locked(name).to_dict()

    def is_active(self, name: str) -> bool:
        with self._lock:
            return bool(self._status_for_locked(name).active)

    def should_stop(self, name: str) -> bool:
        with self._lock:
            return self._is_stop_requested_locked(name)

    def activate(self, name: str, *, run_now: bool = False) -> dict[str, object]:
        normalized_name = name.strip()
        if not normalized_name:
            raise RuntimeError("Job name is required.")

        should_run_now = False
        with self._lock:
            registration = self._registrations.get(normalized_name)
            if registration is None:
                raise RuntimeError(f"Job is not registered: {normalized_name}")

            state = self._status_for_locked(normalized_name)
            state.active = True
            if state.status not in {"searching", "replying", "sending", "stopping"}:
                state.status = "idle"
            state.last_error = None
            self._stop_requested.discard(normalized_name)
            self._append_log_locked(normalized_name, "Job activated.")
            self._schedule_locked(normalized_name)
            should_run_now = run_now and normalized_name not in self._running
            if should_run_now:
                self._append_log_locked(normalized_name, "Run triggered by activation.")
            payload = state.to_dict()

        if should_run_now:
            Thread(target=self._run_cycle, args=(normalized_name, "activation", False), daemon=True).start()
        return payload

    def deactivate(self, name: str, *, stop_processing: bool = True) -> dict[str, object]:
        normalized_name = name.strip()
        if not normalized_name:
            return JobRunStatus(name="", active=False, status="inactive").to_dict()

        with self._lock:
            state = self._status_for_locked(normalized_name)
            state.active = False
            self._remove_schedule_locked(normalized_name)
            if normalized_name in self._running:
                if stop_processing:
                    self._stop_requested.add(normalized_name)
                    state.status = "stopping"
                    self._append_log_locked(normalized_name, "Job deactivated. Stop requested for active run.")
                else:
                    self._append_log_locked(normalized_name, "Job deactivated while run is active.")
            else:
                state.status = "inactive"
                state.finished_at = _now_iso()
                self._append_log_locked(normalized_name, "Job deactivated.")
            return state.to_dict()

    def start(self, name: str, *, run_now: bool = True) -> dict[str, object]:
        normalized_name = name.strip()
        if not normalized_name:
            raise RuntimeError("Job name is required.")

        with self._lock:
            registration = self._registrations.get(normalized_name)
            if registration is None:
                raise RuntimeError(f"Job is not registered: {normalized_name}")

            state = self._status_for_locked(normalized_name)
            state.last_error = None
            self._stop_requested.discard(normalized_name)
            should_run_now = run_now and normalized_name not in self._running
            if should_run_now:
                self._append_log_locked(normalized_name, "Manual run triggered.")
            else:
                self._append_log_locked(normalized_name, "Run already in progress.")
            payload = state.to_dict()

        if should_run_now:
            Thread(target=self._run_cycle, args=(normalized_name, "manual", False), daemon=True).start()
        return payload

    def stop(self, name: str) -> dict[str, object]:
        normalized_name = name.strip()
        if not normalized_name:
            return JobRunStatus(name="", active=False, status="inactive").to_dict()

        with self._lock:
            state = self._status_for_locked(normalized_name)
            if normalized_name in self._running:
                self._stop_requested.add(normalized_name)
                state.status = "stopping"
                self._append_log_locked(normalized_name, "Stop requested for active run.")
            else:
                self._append_log_locked(normalized_name, "Stop requested but no run is active.")
            return state.to_dict()

    def restart(self, name: str) -> dict[str, object]:
        with self._lock:
            self._append_log_locked(name.strip(), "Job restart requested.")
        self.stop(name)
        return self.start(name, run_now=True)

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)

    def _status_for_locked(self, name: str) -> JobRunStatus:
        normalized_name = name.strip()
        if normalized_name not in self._states:
            self._states[normalized_name] = JobRunStatus(name=normalized_name)
        return self._states[normalized_name]

    def _job_id(self, name: str) -> str:
        return f"reddrop::{name}"

    def _remove_schedule_locked(self, name: str) -> None:
        job = self._scheduler.get_job(self._job_id(name))
        if job is not None:
            self._scheduler.remove_job(self._job_id(name))

    def _schedule_locked(self, name: str) -> None:
        registration = self._registrations.get(name)
        if registration is None:
            return
        self._scheduler.add_job(
            self._scheduled_trigger,
            trigger="interval",
            minutes=max(1, int(registration.interval_minutes)),
            id=self._job_id(name),
            replace_existing=True,
            args=[name],
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
        )

    def _set_phase(self, name: str, phase: str) -> None:
        with self._lock:
            state = self._status_for_locked(name)
            if self._is_stop_requested_locked(name):
                state.status = "stopping"
                return
            if state.status != phase:
                self._append_log_locked(name, f"Phase: {phase}.")
            state.status = phase

    def _scheduled_trigger(self, name: str) -> None:
        with self._lock:
            if not self._status_for_locked(name).active:
                return
            self._append_log_locked(name, "Scheduled run triggered.")
        Thread(target=self._run_cycle, args=(name, "scheduled", True), daemon=True).start()

    def _run_cycle(self, name: str, trigger: str = "manual", require_active: bool = True) -> None:
        with self._lock:
            state = self._status_for_locked(name)
            if require_active and not state.active:
                return
            if name in self._running:
                return
            registration = self._registrations.get(name)
            if registration is None:
                state.status = "failed"
                state.last_error = f"Job is not registered: {name}"
                state.finished_at = _now_iso()
                self._append_log_locked(name, f"Run failed: {state.last_error}")
                return
            self._running.add(name)
            state.started_at = _now_iso()
            state.finished_at = None
            state.last_error = None
            self._append_log_locked(name, f"Run started ({trigger}).")
            if self._is_stop_requested_locked(name):
                state.finished_at = _now_iso()
                state.status = "inactive" if not state.active else "idle"
                state.last_run_status = "stopped"
                self._append_log_locked(name, "Run stopped before execution started.")
                self._running.discard(name)
                self._stop_requested.discard(name)
                return

        try:
            output = registration.runner(lambda phase: self._set_phase(name, phase))
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                state = self._status_for_locked(name)
                state.finished_at = _now_iso()
                is_stopped = self._is_stop_requested_locked(name) or "stopped by user request" in str(exc).lower()
                if is_stopped:
                    state.last_error = None
                    state.status = "inactive" if not state.active else "idle"
                    state.last_run_status = "stopped"
                    self._append_log_locked(name, "Run stopped by user request.")
                else:
                    state.last_error = str(exc)
                    if state.active:
                        state.status = "failed"
                    else:
                        state.status = "inactive"
                    state.last_run_status = "failed"
                    self._append_log_locked(name, f"Run failed: {state.last_error}")
        else:
            with self._lock:
                state = self._status_for_locked(name)
                state.finished_at = _now_iso()
                was_stopped = self._is_stop_requested_locked(name)
                if isinstance(output, str) and output.strip():
                    state.last_output = output.strip()[-2000:]
                    self._append_log_locked(name, f"Run output: {state.last_output}")
                if was_stopped:
                    state.status = "inactive" if not state.active else "idle"
                    state.last_run_status = "stopped"
                    self._append_log_locked(name, "Run stopped by user request.")
                else:
                    if state.active:
                        state.status = "idle"
                    else:
                        state.status = "inactive"
                    state.last_run_status = "success"
                    state.successful_runs += 1
                    self._append_log_locked(name, "Run completed successfully.")
        finally:
            with self._lock:
                self._running.discard(name)
                self._stop_requested.discard(name)

    def _append_log_locked(self, name: str, message: str) -> None:
        normalized_name = name.strip()
        if not normalized_name:
            return
        state = self._status_for_locked(normalized_name)
        state.logs.append(RuntimeLogEntry(at=_now_iso(), message=message))
        if len(state.logs) > self.MAX_LOGS_PER_JOB:
            state.logs = state.logs[-self.MAX_LOGS_PER_JOB :]

    def _is_stop_requested_locked(self, name: str) -> bool:
        normalized_name = name.strip()
        return bool(normalized_name and normalized_name in self._stop_requested)
