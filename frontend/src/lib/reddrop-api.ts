export type JobType = 'search' | 'reply'

export type Job = {
  id: string
  name: string
  job_type: JobType
  source_job_id?: string | null
  topic: string
  min_similarity_score: number
  active: boolean
  time_filter: string
  subreddit_limit: number
  threads_limit: number
  replies_per_iteration: number
  max_runtime_minutes: number
  personas: string[]
  last_run_status?: string
  successful_runs?: number
  threads_found_total?: number
  threads_replied_total?: number
  runtime?: JobRuntime
}

export type JobRuntime = {
  id?: string
  name: string
  active?: boolean
  status: string
  last_run_status?: string
  successful_runs?: number
  started_at?: string | null
  finished_at?: string | null
  last_error?: string | null
  last_output?: string | null
  logs?: JobRuntimeLog[]
}

export type JobRuntimeLog = {
  at: string
  message: string
}

export type ThreadItem = {
  job_id: string
  job_name: string
  topic: string
  thread_id: string
  title: string
  subreddit: string
  url: string
  score: number
  num_comments: number
  created_utc?: string | number | null
  selftext: string
  semantic_similarity: number
  user_has_commented: boolean
  reply: string
  has_reply: boolean
  search_artifact?: string
  search_updated_at?: string
}

export type ThreadListResponse = {
  threads: ThreadItem[]
  total: number
  limit: number
  offset: number
}

export type UpsertJobPayload = {
  name: string
  job_type: JobType
  source_job_id?: string | null
  topic: string
  min_similarity_score: number
  active: boolean
  time_filter: string
  subreddit_limit: number
  threads_limit: number
  replies_per_iteration: number
  max_runtime_minutes: number
  personas: string[]
}

export type PersonaProfile = {
  name: string
  description: string
  objective: string
}

export type PersonasResponse = {
  personas: PersonaProfile[]
  source?: string | null
}

export type UpsertPersonasPayload = {
  personas: PersonaProfile[]
}

export type RedditSettings = {
  client_id: string
  client_secret: string
  user_agent: string
  username: string
  password: string
}

export type OpenRouterSettings = {
  api_key: string
  base_url: string
  model: string
  http_referer: string
  x_title: string
  timeout_seconds: string
}

export type SettingsPayload = {
  reddit: RedditSettings
  openrouter: OpenRouterSettings
}

export type ThreadListParams = {
  job_id?: string
  job_name?: string
  subreddit?: string
  min_similarity?: number
  only_open?: boolean
  has_reply?: boolean
  limit?: number
  offset?: number
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = (await response.json()) as { detail?: string }
      if (payload?.detail) detail = payload.detail
    } catch {
      // keep fallback status text
    }
    throw new Error(detail)
  }

  return (await response.json()) as T
}

function queryFrom(params: ThreadListParams): string {
  const query = new URLSearchParams()
  if (params.job_id) query.set('job_id', params.job_id)
  if (params.job_name) query.set('job_name', params.job_name)
  if (params.subreddit) query.set('subreddit', params.subreddit)
  if (typeof params.min_similarity === 'number') query.set('min_similarity', String(params.min_similarity))
  if (typeof params.only_open === 'boolean') query.set('only_open', String(params.only_open))
  if (typeof params.has_reply === 'boolean') query.set('has_reply', String(params.has_reply))
  if (typeof params.limit === 'number') query.set('limit', String(params.limit))
  if (typeof params.offset === 'number') query.set('offset', String(params.offset))
  const suffix = query.toString()
  return suffix ? `?${suffix}` : ''
}

export const reddropApi = {
  listJobs: () => request<{ jobs: Job[] }>('/jobs'),
  createJob: (payload: UpsertJobPayload) =>
    request<Job>('/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  deleteJob: (jobRef: string) =>
    request<{ deleted: boolean; id: string; name: string }>(`/jobs/${encodeURIComponent(jobRef)}`, {
      method: 'DELETE',
    }),
  startJob: (jobRef: string) =>
    request<JobRuntime>(`/jobs/${encodeURIComponent(jobRef)}/start`, { method: 'POST' }),
  stopJob: (jobRef: string) =>
    request<JobRuntime>(`/jobs/${encodeURIComponent(jobRef)}/stop`, { method: 'POST' }),
  activateJob: (jobRef: string) =>
    request<JobRuntime>(`/jobs/${encodeURIComponent(jobRef)}/activate`, { method: 'POST' }),
  deactivateJob: (jobRef: string) =>
    request<JobRuntime>(`/jobs/${encodeURIComponent(jobRef)}/deactivate`, { method: 'POST' }),
  listThreads: (params: ThreadListParams = {}) =>
    request<ThreadListResponse>(`/threads${queryFrom(params)}`),
  createThreadReply: (
    jobRef: string,
    subreddit: string,
    threadId: string,
    persona: string,
    reply?: string
  ) =>
    request<ThreadItem>(`/threads/${encodeURIComponent(jobRef)}/${encodeURIComponent(subreddit)}/${encodeURIComponent(threadId)}/reply`, {
      method: 'POST',
      body: JSON.stringify(
        typeof reply === 'string' && reply.trim().length > 0
          ? { persona, reply }
          : { persona }
      ),
    }),
  sendThreadReply: (jobRef: string, subreddit: string, threadId: string) =>
    request<ThreadItem>(`/threads/${encodeURIComponent(jobRef)}/${encodeURIComponent(subreddit)}/${encodeURIComponent(threadId)}/send`, {
      method: 'POST',
    }),
  listPersonas: () => request<PersonasResponse>('/personas'),
  savePersonas: (payload: UpsertPersonasPayload) =>
    request<PersonasResponse>('/personas', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  listSettings: () => request<SettingsPayload>('/settings'),
  saveSettings: (payload: SettingsPayload) =>
    request<SettingsPayload>('/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
}
