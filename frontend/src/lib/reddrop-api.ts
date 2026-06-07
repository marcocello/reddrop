export type JobType = 'search' | 'reply'
export type SocialPlatform = 'reddit' | 'linkedin'

export type SearchPlan = {
  topics?: string[]
  subreddits?: string[]
  queries?: string[]
  query?: string | null
}

export type SearchQuery = {
  id: string
  name: string
  social: SocialPlatform
  topic: string
  plan: SearchPlan
  status: 'draft' | 'saved'
  version: number
  created_at: string
  updated_at: string
  saved_at?: string | null
}

export type Job = {
  id: string
  name: string
  job_type: JobType
  social: SocialPlatform
  auth_id?: string | null
  search_query_id?: string | null
  search_query_version?: number | null
  source_job_id?: string | null
  topic?: string
  min_similarity_score?: number
  active: boolean
  time_filter?: string
  subreddit_limit?: number
  threads_limit?: number
  scroll_rounds?: number
  sort_by?: string
  selector_wait_timeout_seconds?: number
  detail_wait_timeout_seconds?: number
  max_comment_expand_clicks?: number
  detail_enrich_top_k?: number
  search_plan?: SearchPlan
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
  conversation_id: string
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
  published_at?: string | null
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

export type DeleteThreadResponse = {
  deleted: boolean
  conversation_id: string
  job_id: string
  thread_id: string
  subreddit: string
  search_file: string
  updated_at: string
}

export type UpsertJobPayload = {
  name: string
  job_type: JobType
  social: SocialPlatform
  auth_id?: string | null
  search_query_id?: string | null
  search_query_version?: number | null
  source_job_id?: string | null
  topic?: string
  min_similarity_score?: number
  active: boolean
  time_filter?: string
  subreddit_limit?: number
  threads_limit?: number
  scroll_rounds?: number
  sort_by?: string
  selector_wait_timeout_seconds?: number
  detail_wait_timeout_seconds?: number
  max_comment_expand_clicks?: number
  detail_enrich_top_k?: number
  search_plan?: SearchPlan
  replies_per_iteration?: number
  max_runtime_minutes: number
  personas?: string[]
}

export type PersonaProfile = {
  name: string
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
  linkedin_username: string
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

export type SearchQueriesResponse = {
  search_queries: SearchQuery[]
}

export type UpsertSearchQueryPayload = {
  name: string
  social: SocialPlatform
  topic: string
  plan: SearchPlan
}

export type GenerateSearchQueryPayload = {
  name?: string
  social: SocialPlatform
  topic: string
  subreddit_limit?: number
}

export type SearchQueryResponse = {
  search_query: SearchQuery
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
  listSearchQueries: () => request<SearchQueriesResponse>('/search-queries'),
  createSearchQuery: (payload: UpsertSearchQueryPayload) =>
    request<SearchQueryResponse>('/search-queries', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateSearchQuery: (searchQueryRef: string, payload: UpsertSearchQueryPayload) =>
    request<SearchQueryResponse>(`/search-queries/${encodeURIComponent(searchQueryRef)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  generateSearchQuery: (payload: GenerateSearchQueryPayload) =>
    request<SearchQueryResponse>('/search-queries/generate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  saveSearchQuery: (searchQueryRef: string) =>
    request<SearchQueryResponse>(`/search-queries/${encodeURIComponent(searchQueryRef)}/save`, {
      method: 'POST',
    }),
  deleteSearchQuery: (searchQueryRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/search-queries/${encodeURIComponent(searchQueryRef)}`, {
      method: 'DELETE',
    }),
  deleteSearchQueryCascade: (searchQueryRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/search-queries/${encodeURIComponent(searchQueryRef)}?cascade=true`, {
      method: 'DELETE',
    }),
  deleteJob: (jobRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/jobs/${encodeURIComponent(jobRef)}`, {
      method: 'DELETE',
    }),
  deleteJobCascade: (jobRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/jobs/${encodeURIComponent(jobRef)}?cascade=true`, {
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
  deleteThread: (jobRef: string, subreddit: string, threadId: string) =>
    request<DeleteThreadResponse>(`/threads/${encodeURIComponent(jobRef)}/${encodeURIComponent(subreddit)}/${encodeURIComponent(threadId)}`, {
      method: 'DELETE',
    }),
  listPersonas: () => request<PersonasResponse>('/personas'),
  savePersonas: (payload: UpsertPersonasPayload) =>
    request<PersonasResponse>('/personas', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deletePersona: (personaRef: string) =>
    request<{ deleted: boolean; name: string; cascade?: boolean }>(`/personas/${encodeURIComponent(personaRef)}`, {
      method: 'DELETE',
    }),
  deletePersonaCascade: (personaRef: string) =>
    request<{ deleted: boolean; name: string; cascade?: boolean }>(`/personas/${encodeURIComponent(personaRef)}?cascade=true`, {
      method: 'DELETE',
    }),
  listAuths: () => request<AuthsResponse>('/auths'),
  saveAuths: (payload: UpsertAuthsPayload) =>
    request<AuthsResponse>('/auths', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteAuth: (authRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/auths/${encodeURIComponent(authRef)}`, {
      method: 'DELETE',
    }),
  deleteAuthCascade: (authRef: string) =>
    request<{ deleted: boolean; id: string; name: string; cascade?: boolean }>(`/auths/${encodeURIComponent(authRef)}?cascade=true`, {
      method: 'DELETE',
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
