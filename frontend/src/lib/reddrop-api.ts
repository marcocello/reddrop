export type JobType = 'search' | 'reply'
export type SocialPlatform = 'reddit' | 'linkedin'

export type Job = {
  id: string
  name: string
  job_type: JobType
  social: SocialPlatform
  auth_id?: string | null
  source_job_id?: string | null
  topic?: string
  min_similarity_score?: number
  active: boolean
  time_filter?: string
  subreddit_limit?: number
  threads_limit?: number
  replies_per_iteration?: number
  max_runtime_minutes: number
  personas?: string[]
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
  social?: SocialPlatform
  topic: string
  search_job_id?: string
  search_job_name?: string
  reply_job_id?: string | null
  reply_job_name?: string | null
  linked_reply_job_ids?: string[]
  linked_reply_job_names?: string[]
  reply_personas?: string[]
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
  social: SocialPlatform
  auth_id?: string | null
  source_job_id?: string | null
  topic?: string
  min_similarity_score?: number
  active: boolean
  time_filter?: string
  subreddit_limit?: number
  threads_limit?: number
  replies_per_iteration?: number
  max_runtime_minutes: number
  personas?: string[]
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

export type AuthProfile = {
  id: string
  name: string
  social: SocialPlatform
  status?: 'ok' | 'invalid'
  reddit_client_id: string
  reddit_client_secret: string
  reddit_user_agent: string
  reddit_username: string
  reddit_password: string
  linkedin_email: string
  linkedin_password: string
  linkedin_headless: string
  linkedin_user_data_dir: string
  linkedin_wait_seconds: string
}

export type AuthsResponse = {
  auths: AuthProfile[]
  source?: string | null
}

export type AuthValidationResult = {
  auth_id: string
  social: SocialPlatform
  status: 'ok' | 'invalid'
}

export type AuthValidationResultsResponse = {
  results: AuthValidationResult[]
}

export type UpsertAuthsPayload = {
  auths: AuthProfile[]
}

export type OpenRouterSettings = {
  api_key: string
  base_url: string
  model: string
  http_referer: string
  x_title: string
  timeout_seconds: string
}

export type LinkedinLoginResponse = {
  status: 'session_saved'
  auth_id?: string
  user_data_dir: string
  login_url: string
}

export type SettingsPayload = {
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
  listAuths: () => request<AuthsResponse>('/auths'),
  saveAuths: (payload: UpsertAuthsPayload) =>
    request<AuthsResponse>('/auths', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  validateAuth: (authRef: string) =>
    request<AuthValidationResult>(`/auths/${encodeURIComponent(authRef)}/validate`, {
      method: 'POST',
    }),
  validateAuths: () =>
    request<AuthValidationResultsResponse>('/auths/validate', {
      method: 'POST',
    }),
  runLinkedinAuthLogin: (authRef: string) =>
    request<LinkedinLoginResponse>(`/auths/${encodeURIComponent(authRef)}/linkedin/login`, {
      method: 'POST',
    }),
  listSettings: () => request<SettingsPayload>('/settings'),
  saveSettings: (payload: SettingsPayload) =>
    request<SettingsPayload>('/settings', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
}
