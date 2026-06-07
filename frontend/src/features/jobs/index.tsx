import { useCallback, useEffect, useMemo, useState } from 'react'
import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  type ColumnDef,
  type ColumnOrderState,
  type ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFacetedRowModel,
  getFacetedUniqueValues,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type FilterFn,
  type RowSelectionState,
  type SortingState,
  useReactTable,
  type VisibilityState,
} from '@tanstack/react-table'
import { ChevronDown, Copy, Eye, MessageSquareReply, MoreHorizontal, Play, Plus, Power, PowerOff, RefreshCw, Search, Square, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { DataTableColumnHeader, DataTablePagination, DataTableToolbar } from '@/components/data-table'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { SelectionActionBar } from '@/components/selection-action-bar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader as ModalDialogHeader,
  DialogTitle as ModalDialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import {
  reddropApi,
  type AuthProfile,
  type Job,
  type JobType,
  type PersonaProfile,
  type SearchQuery,
  type SocialPlatform,
  type UpsertJobPayload,
} from '@/lib/reddrop-api'

const ACTIONS_COLUMN_ID = 'actions'
const SELECT_COLUMN_ID = 'select'
const AUTO_REFRESH_OPTIONS = [
  { value: 0, label: 'Off', menuLabel: 'Auto-refresh off' },
  { value: 30, label: 'Every 30s', menuLabel: 'Every 30s' },
  { value: 60, label: 'Every 1m', menuLabel: 'Every 1m' },
  { value: 300, label: 'Every 5m', menuLabel: 'Every 5m' },
  { value: 900, label: 'Every 15m', menuLabel: 'Every 15m' },
] as const
const RUN_FREQUENCY_OPTIONS = [
  { value: 60, label: 'Every 1 hour' },
  { value: 360, label: 'Every 6 hours' },
  { value: 720, label: 'Every 12 hours' },
  { value: 1440, label: 'Every 24 hours' },
] as const
const DEFAULT_RUN_FREQUENCY_MINUTES = 1440

function buildInitialForm(): UpsertJobPayload {
  return {
    name: '',
    job_type: 'search',
    social: 'reddit',
    auth_id: null,
    search_query_id: null,
    search_query_version: null,
    source_job_id: null,
    topic: '',
    min_similarity_score: 0.35,
    active: false,
    time_filter: 'week',
    subreddit_limit: 5,
    threads_limit: 10,
    scroll_rounds: 15,
    sort_by: 'relevance',
    selector_wait_timeout_seconds: 20,
    detail_wait_timeout_seconds: 8,
    max_comment_expand_clicks: 30,
    detail_enrich_top_k: 12,
    search_plan: undefined,
    replies_per_iteration: 3,
    max_runtime_minutes: DEFAULT_RUN_FREQUENCY_MINUTES,
    personas: [],
  }
}

const arrayFilterFn: FilterFn<Job> = (row, columnId, value) => {
  const selected = Array.isArray(value) ? (value as string[]) : []
  if (selected.length === 0) {
    return true
  }
  return selected.includes(String(row.getValue(columnId)))
}

function formatWhen(value?: string | null): string {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

function formatStatusLabel(status: string | undefined): string {
  const value = (status ?? 'idle').trim()
  if (!value) return 'Idle'
  return value.charAt(0).toUpperCase() + value.slice(1)
}

function statusBadgeClass(status: string | undefined): string {
  if (status === 'searching') return 'bg-blue-600 text-white'
  if (status === 'replying') return 'bg-cyan-600 text-white'
  if (status === 'sending') return 'bg-emerald-600 text-white'
  if (status === 'running') return 'bg-blue-600 text-white'
  if (status === 'idle') return 'bg-secondary text-secondary-foreground'
  if (status === 'failed') return 'bg-red-600 text-white'
  if (status === 'stopping') return 'bg-amber-500 text-black'
  if (status === 'inactive') return 'bg-muted text-muted-foreground'
  return 'bg-muted text-muted-foreground'
}

function activeBadgeClass(active: boolean): string {
  return active ? 'bg-emerald-600 text-white' : 'bg-muted text-muted-foreground'
}

function formatJobType(value: JobType): string {
  return value === 'reply' ? 'Reply' : 'Search'
}

function jobTypeIcon(value: JobType) {
  return value === 'reply' ? MessageSquareReply : Search
}

function formatSocial(value: SocialPlatform): string {
  return value === 'linkedin' ? 'LinkedIn' : 'Reddit'
}

function socialIcon(platform: SocialPlatform): IconDefinition {
  return platform === 'linkedin' ? faLinkedinIn : faRedditAlien
}

function jobTypeBadgeClass(_value: JobType): string {
  return 'bg-muted text-muted-foreground'
}

function displayJobTopic(job: Pick<Job, 'job_type' | 'topic'>): string {
  if (job.job_type === 'reply') return 'N/A'
  const normalized = (job.topic ?? '').trim()
  return normalized || '-'
}

function shortId(value: string): string {
  const normalized = value.trim()
  if (!normalized) return '-'
  return normalized.slice(0, 8)
}

function authDisplayName(auth: AuthProfile): string {
  if (auth.social === 'reddit') {
    const username = auth.reddit_username.trim()
    if (username) return username
  }
  if (auth.social === 'linkedin') {
    const username = auth.linkedin_username.trim()
    if (username) return username
  }
  const fallbackName = auth.name?.trim()
  if (fallbackName) return fallbackName
  return shortId(auth.id)
}

function buildDuplicateJobName(baseName: string, existingNames: Set<string>): string {
  const normalized = baseName.trim().replace(/\s+/g, '-') || 'job'
  let attempt = `${normalized}-copy`
  let index = 2
  while (existingNames.has(attempt.toLowerCase())) {
    attempt = `${normalized}-copy-${index}`
    index += 1
  }
  return attempt
}

function normalizeNameToken(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
  return normalized || 'job'
}

function buildReplySuffix(sourceSearchJobName?: string): string {
  if (!sourceSearchJobName) return 'reply_'
  return `reply_${normalizeNameToken(sourceSearchJobName)}`
}

function extractReplyPrefix(name: string, sourceSearchJobName?: string): string {
  const normalized = normalizeNameToken(name)
  const suffix = buildReplySuffix(sourceSearchJobName)
  if (!normalized || normalized === suffix) return ''
  const suffixToken = `_${suffix}`
  if (normalized.endsWith(suffixToken)) {
    return normalized.slice(0, -suffixToken.length)
  }
  return normalized
}

function buildReplyJobName(prefix: string, sourceSearchJobName?: string): string {
  const suffix = buildReplySuffix(sourceSearchJobName)
  const normalizedPrefix = extractReplyPrefix(prefix, sourceSearchJobName)
  if (!normalizedPrefix) return suffix
  return `${normalizedPrefix}_${suffix}`
}

function normalizeRunFrequencyMinutes(value: number): number {
  const options = new Set<number>(RUN_FREQUENCY_OPTIONS.map((item) => item.value))
  return options.has(value) ? value : DEFAULT_RUN_FREQUENCY_MINUTES
}

function formatRunFrequency(minutes: number): string {
  const match = RUN_FREQUENCY_OPTIONS.find((item) => item.value === minutes)
  if (match) return match.label
  if (minutes % 60 === 0) {
    const hours = minutes / 60
    return `Every ${hours} hour${hours === 1 ? '' : 's'}`
  }
  return `Every ${minutes} minutes`
}

export function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [autoRefreshSeconds, setAutoRefreshSeconds] = useState<number>(30)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([{ id: 'name', desc: false }])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([])
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [draggingColumnId, setDraggingColumnId] = useState<string | null>(null)

  const [personaOptions, setPersonaOptions] = useState<PersonaProfile[]>([])
  const [authOptions, setAuthOptions] = useState<AuthProfile[]>([])
  const [searchQueryOptions, setSearchQueryOptions] = useState<SearchQuery[]>([])

  const [sheetOpen, setSheetOpen] = useState(false)
  const [sheetTab, setSheetTab] = useState<'overview' | 'activity'>('overview')
  const [jobActivityFilter, setJobActivityFilter] = useState<'all' | 'errors' | 'warnings'>('all')
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [pendingDeleteJob, setPendingDeleteJob] = useState<Job | null>(null)
  const [pendingDeleteJobs, setPendingDeleteJobs] = useState<Job[]>([])
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [form, setForm] = useState<UpsertJobPayload>(buildInitialForm)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const payload = await reddropApi.listJobs()
      setJobs(payload.jobs)
      setSelectedJob((prev) => {
        if (!prev) return null
        return payload.jobs.find((item) => item.id === prev.id) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load jobs'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  const loadPersonas = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listPersonas()
      setPersonaOptions(payload.personas)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load personas'
      toast.error(message)
    }
  }, [])

  const loadAuths = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listAuths()
      setAuthOptions(payload.auths)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load auths'
      toast.error(message)
    }
  }, [])

  const loadSearchQueries = useCallback(async (): Promise<void> => {
    try {
      const payload = await reddropApi.listSearchQueries()
      setSearchQueryOptions(payload.search_queries)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load search queries'
      toast.error(message)
    }
  }, [])

  useEffect(() => {
    void load()
    void loadPersonas()
    void loadAuths()
    void loadSearchQueries()
  }, [load, loadAuths, loadPersonas, loadSearchQueries])

  useEffect(() => {
    if (autoRefreshSeconds <= 0) {
      return
    }
    const timer = window.setInterval(() => {
      void load()
    }, autoRefreshSeconds * 1000)
    return () => window.clearInterval(timer)
  }, [autoRefreshSeconds, load])

  const autoRefreshLabel = useMemo(() => {
    const selected = AUTO_REFRESH_OPTIONS.find((option) => option.value === autoRefreshSeconds)
    return selected?.label ?? 'Off'
  }, [autoRefreshSeconds])

  const statusFilterOptions = useMemo(() => {
    const statuses = new Set<string>()
    jobs.forEach((job) => statuses.add(job.runtime?.status ?? 'idle'))
    return Array.from(statuses)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: formatStatusLabel(value), value }))
  }, [jobs])

  const activeFilterOptions = useMemo(() => {
    const states = new Set<string>()
    jobs.forEach((job) => states.add(job.active ? 'active' : 'inactive'))
    return Array.from(states)
      .sort((left, right) => left.localeCompare(right))
      .map((value) => ({ label: formatStatusLabel(value), value }))
  }, [jobs])

  const jobTypeFilterOptions = useMemo(() => {
    const values = new Set<JobType>()
    jobs.forEach((job) => values.add(job.job_type))
    return Array.from(values).map((value) => ({
      label: formatJobType(value),
      value,
    }))
  }, [jobs])

  const searchingJobOptions = useMemo(
    () => jobs.filter((job) => job.job_type === 'search'),
    [jobs]
  )

  const sourceSearchNameById = useMemo(() => {
    const map = new Map<string, string>()
    searchingJobOptions.forEach((job) => map.set(job.id, job.name))
    return map
  }, [searchingJobOptions])

  const sourceSearchSocialById = useMemo(() => {
    const map = new Map<string, SocialPlatform>()
    searchingJobOptions.forEach((job) => map.set(job.id, job.social))
    return map
  }, [searchingJobOptions])

  const sourceSearchAuthById = useMemo(() => {
    const map = new Map<string, string | null>()
    searchingJobOptions.forEach((job) => map.set(job.id, job.auth_id ?? null))
    return map
  }, [searchingJobOptions])

  const authNameById = useMemo(() => {
    const map = new Map<string, string>()
    authOptions.forEach((auth) => map.set(auth.id, authDisplayName(auth)))
    return map
  }, [authOptions])

  const searchPathById = useMemo(() => {
    const map = new Map<string, SearchQuery>()
    searchQueryOptions.forEach((item) => map.set(item.id, item))
    return map
  }, [searchQueryOptions])

  const openCreateDialog = useCallback((): void => {
    setSelectedJob(null)
    setForm(buildInitialForm())
    setSheetOpen(false)
    setCreateDialogOpen(true)
  }, [])

  const openJobSheet = useCallback((job: Job): void => {
    setSelectedJob(job)
    setSheetTab('overview')
    setJobActivityFilter('all')
    setSheetOpen(true)
  }, [])

  const createJob = useCallback(async (): Promise<void> => {
    const replyPersonas = form.personas ?? []
    const primaryReplyPersona = replyPersonas[0] ?? ''
    const selectedSearchQuery = form.search_query_id ? searchPathById.get(form.search_query_id) : null
    if (form.job_type === 'search' && !form.auth_id) {
      toast.error('Auth profile is required for search jobs')
      return
    }
    if (form.job_type === 'search' && !form.search_query_id) {
      toast.error('Search query is required for search jobs')
      return
    }
    if (form.job_type === 'search' && !selectedSearchQuery) {
      toast.error('Selected search query is not available')
      return
    }
    if (form.job_type === 'search' && selectedSearchQuery?.status !== 'saved') {
      toast.error('Search query must be saved before creating a search job')
      return
    }
    if (form.job_type === 'reply' && !form.source_job_id) {
      toast.error('Source search job is required for reply jobs')
      return
    }
    if (form.job_type === 'reply' && replyPersonas.length === 0) {
      toast.error('At least one persona is required for reply jobs')
      return
    }

    const sourceName =
      form.source_job_id ? sourceSearchNameById.get(form.source_job_id) : undefined
    const sourceSocial =
      form.source_job_id ? sourceSearchSocialById.get(form.source_job_id) : undefined
    const sourceAuthId =
      form.source_job_id ? sourceSearchAuthById.get(form.source_job_id) : undefined
    let normalizedName = form.name.trim()
    if (form.job_type === 'reply') {
      normalizedName = buildReplyJobName(normalizedName, sourceName)
    }
    if (!normalizedName) {
      toast.error('Job name is required')
      return
    }

    const payload: UpsertJobPayload =
      form.job_type === 'reply'
        ? {
            name: normalizedName,
            job_type: 'reply',
            social: sourceSocial ?? form.social,
            auth_id: sourceAuthId ?? form.auth_id ?? null,
            source_job_id: form.source_job_id,
            min_similarity_score: form.min_similarity_score,
            replies_per_iteration: form.replies_per_iteration,
            max_runtime_minutes: form.max_runtime_minutes,
            personas: [primaryReplyPersona],
            active: form.active,
          }
        : {
            name: normalizedName,
            job_type: 'search',
            social: selectedSearchQuery?.social ?? form.social,
            auth_id: form.auth_id,
            search_query_id: selectedSearchQuery?.id ?? null,
            search_query_version: selectedSearchQuery?.version ?? null,
            threads_limit: form.threads_limit,
            max_runtime_minutes: form.max_runtime_minutes,
            active: form.active,
            ...(selectedSearchQuery?.social === 'reddit'
              ? {
                  time_filter: form.time_filter,
                  subreddit_limit: form.subreddit_limit,
                }
              : {
                  time_filter: form.time_filter,
                  scroll_rounds: form.scroll_rounds,
                  sort_by: form.sort_by,
                  selector_wait_timeout_seconds: form.selector_wait_timeout_seconds,
                  detail_wait_timeout_seconds: form.detail_wait_timeout_seconds,
                  max_comment_expand_clicks: form.max_comment_expand_clicks,
                  detail_enrich_top_k: form.detail_enrich_top_k,
                }),
          }

    setSaving(true)
    try {
      await reddropApi.createJob(payload)
      toast.success(`Created ${formatJobType(payload.job_type).toLowerCase()} job ${payload.name}`)
      await load()
      setCreateDialogOpen(false)
      setSelectedJob(null)
      setForm(buildInitialForm())
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create job'
      toast.error(message)
    } finally {
      setSaving(false)
    }
  }, [form, load, searchPathById, sourceSearchAuthById, sourceSearchNameById, sourceSearchSocialById])

  const runAction = useCallback(async (job: Job, action: 'start' | 'stop'): Promise<void> => {
    const key = `${job.id}:${action}`
    setBusyKey(key)
    try {
      if (action === 'start') await reddropApi.startJob(job.id)
      if (action === 'stop') await reddropApi.stopJob(job.id)
      toast.success(`Job ${job.name}: ${action} run requested`)
      await load()
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to ${action} ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load])

  const activationAction = useCallback(async (job: Job, action: 'activate' | 'deactivate'): Promise<void> => {
    const key = `${job.id}:${action}`
    setBusyKey(key)
    try {
      if (action === 'activate') await reddropApi.activateJob(job.id)
      if (action === 'deactivate') await reddropApi.deactivateJob(job.id)
      toast.success(`Job ${job.name}: ${action} requested`)
      await load()
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to ${action} ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load])

  const deleteJob = useCallback(async (job: Job, cascade: boolean): Promise<void> => {
    const key = `${job.id}:delete`
    setBusyKey(key)
    try {
      if (cascade) await reddropApi.deleteJobCascade(job.id)
      else await reddropApi.deleteJob(job.id)
      toast.success(`Deleted job ${job.name}`)
      await load()
      setSelectedJob((prev) => (prev?.id === job.id ? null : prev))
      if (selectedJob?.id === job.id) {
        setSheetOpen(false)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : `Failed to delete ${job.name}`
      toast.error(message)
    } finally {
      setBusyKey(null)
    }
  }, [load, selectedJob?.id])

  const requestDeleteSelectedJobs = useCallback((): void => {
    const selectedIds = new Set(
      Object.entries(rowSelection)
        .filter(([, selected]) => Boolean(selected))
        .map(([id]) => id)
    )
    const selected = jobs.filter((job) => selectedIds.has(job.id))
    if (selected.length === 0) return
    setPendingDeleteJobs(selected)
  }, [jobs, rowSelection])

  const deleteSelectedJobs = useCallback(async (): Promise<void> => {
    if (pendingDeleteJobs.length === 0) return
    setBusyKey('bulk-delete')
    let deletedRootCount = 0
    const failures: string[] = []
    const ordered = [...pendingDeleteJobs].sort((left, right) => {
      if (left.job_type === right.job_type) return left.name.localeCompare(right.name)
      return left.job_type === 'reply' ? -1 : 1
    })
    for (const job of ordered) {
      try {
        if (job.job_type === 'search') await reddropApi.deleteJobCascade(job.id)
        else await reddropApi.deleteJob(job.id)
        deletedRootCount += 1
      } catch (error) {
        const message = error instanceof Error ? error.message : `Failed to delete ${job.name}`
        if (message.toLowerCase().includes('job not found')) {
          continue
        }
        failures.push(`${job.name}: ${message}`)
      }
    }
    setRowSelection({})
    setPendingDeleteJobs([])
    await load()
    if (deletedRootCount > 0) {
      toast.success(`Deleted ${deletedRootCount} job${deletedRootCount === 1 ? '' : 's'}`)
    }
    if (failures.length > 0) {
      toast.error(failures[0])
    }
    setBusyKey(null)
  }, [load, pendingDeleteJobs])

  const openDuplicateJobDialog = useCallback((job: Job): void => {
    const existingNames = new Set(jobs.map((item) => item.name.trim().toLowerCase()))
    let duplicateName = buildDuplicateJobName(job.name, existingNames)
    if (job.job_type === 'reply') {
      const sourceName = job.source_job_id ? sourceSearchNameById.get(job.source_job_id) : undefined
      duplicateName = extractReplyPrefix(duplicateName, sourceName)
    }
    setSelectedJob(null)
    setSheetOpen(false)
    setForm({
      name: duplicateName,
      job_type: job.job_type,
      social: job.social,
      auth_id: job.auth_id ?? null,
      search_query_id: job.search_query_id ?? null,
      search_query_version: job.search_query_version ?? null,
      source_job_id: job.source_job_id ?? null,
      topic: job.job_type === 'reply' ? '' : (job.topic ?? ''),
      search_plan: job.job_type === 'search' ? (job.search_plan ?? undefined) : undefined,
      min_similarity_score: job.min_similarity_score ?? 0.35,
      active: job.active,
      time_filter: job.time_filter ?? 'week',
      subreddit_limit: job.subreddit_limit ?? 5,
      threads_limit: job.threads_limit ?? 10,
      scroll_rounds: job.scroll_rounds ?? 15,
      sort_by: job.sort_by ?? 'relevance',
      selector_wait_timeout_seconds: job.selector_wait_timeout_seconds ?? 20,
      detail_wait_timeout_seconds: job.detail_wait_timeout_seconds ?? 8,
      max_comment_expand_clicks: job.max_comment_expand_clicks ?? 30,
      detail_enrich_top_k: job.detail_enrich_top_k ?? 12,
      replies_per_iteration: job.replies_per_iteration ?? 3,
      max_runtime_minutes: normalizeRunFrequencyMinutes(job.max_runtime_minutes),
      personas:
        job.job_type === 'reply' && (job.personas ?? []).length > 0
          ? [job.personas![0]]
          : [...(job.personas ?? [])],
    })
    setCreateDialogOpen(true)
  }, [jobs, sourceSearchNameById])

  const requestDeleteJob = useCallback((job: Job): void => {
    setPendingDeleteJob(job)
  }, [])

  const pendingDeleteJobRelatedReplies = useMemo(() => {
    if (!pendingDeleteJob || pendingDeleteJob.job_type !== 'search') return [] as Job[]
    return jobs.filter((item) => item.job_type === 'reply' && item.source_job_id === pendingDeleteJob.id)
  }, [jobs, pendingDeleteJob])

  const pendingDeleteJobsRelatedReplies = useMemo(() => {
    if (pendingDeleteJobs.length === 0) return [] as Job[]
    const selectedIds = new Set(pendingDeleteJobs.map((item) => item.id))
    const selectedSearchIds = new Set(
      pendingDeleteJobs.filter((item) => item.job_type === 'search').map((item) => item.id)
    )
    if (selectedSearchIds.size === 0) return [] as Job[]
    const related = jobs.filter(
      (item) =>
        item.job_type === 'reply'
        && Boolean(item.source_job_id)
        && selectedSearchIds.has(String(item.source_job_id))
        && !selectedIds.has(item.id)
    )
    return related
  }, [jobs, pendingDeleteJobs])

  const confirmDeleteJob = useCallback(async (): Promise<void> => {
    if (!pendingDeleteJob) return
    await deleteJob(pendingDeleteJob, pendingDeleteJobRelatedReplies.length > 0)
    setPendingDeleteJob(null)
  }, [deleteJob, pendingDeleteJob, pendingDeleteJobRelatedReplies.length])

  const confirmDeleteSelectedJobs = useCallback(async (): Promise<void> => {
    await deleteSelectedJobs()
  }, [deleteSelectedJobs])

  const columns = useMemo<ColumnDef<Job>[]>(
    () => [
      {
        id: SELECT_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        header: ({ table }) => (
          <Checkbox
            aria-label='Select all jobs'
            checked={
              table.getIsAllPageRowsSelected() ||
              (table.getIsSomePageRowsSelected() && 'indeterminate')
            }
            onCheckedChange={(value) => table.toggleAllPageRowsSelected(Boolean(value))}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            aria-label={`Select job ${row.original.name}`}
            checked={row.getIsSelected()}
            onCheckedChange={(value) => row.toggleSelected(Boolean(value))}
            onClick={(event) => event.stopPropagation()}
          />
        ),
      },
      {
        accessorKey: 'id',
        header: ({ column }) => <DataTableColumnHeader column={column} title='ID' />,
        cell: ({ row }) => (
          <span className='font-mono text-xs' title={row.original.id}>
            {shortId(row.original.id)}
          </span>
        ),
      },
      {
        accessorKey: 'name',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Name' />,
        enableHiding: false,
        cell: ({ row }) => <span className='font-medium'>{row.original.name}</span>,
      },
      {
        id: 'job_type',
        accessorFn: (row) => row.job_type,
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Type' />,
        cell: ({ row }) => {
          const JobTypeIcon = jobTypeIcon(row.original.job_type)
          return (
            <Badge className={jobTypeBadgeClass(row.original.job_type)}>
              <span className='inline-flex items-center gap-1.5'>
                <JobTypeIcon className='size-3.5' />
                {formatJobType(row.original.job_type)}
              </span>
            </Badge>
          )
        },
      },
      {
        id: 'social',
        accessorFn: (row) => row.social,
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Social' />,
        cell: ({ row }) => (
          <Badge className='bg-muted text-muted-foreground'>
            <span className='inline-flex items-center gap-1.5'>
              <FontAwesomeIcon icon={socialIcon(row.original.social)} className='size-3.5' />
              {formatSocial(row.original.social)}
            </span>
          </Badge>
        ),
      },
      {
        id: 'source',
        accessorFn: (row) => {
          if (row.job_type !== 'reply') return '-'
          if (!row.source_job_id) return '-'
          return sourceSearchNameById.get(row.source_job_id) ?? row.source_job_id
        },
        header: ({ column }) => <DataTableColumnHeader column={column} title='Source' />,
        cell: ({ row }) => {
          if (row.original.job_type !== 'reply' || !row.original.source_job_id) {
            return <span className='text-muted-foreground'>-</span>
          }
          const sourceId = row.original.source_job_id
          const sourceName = sourceSearchNameById.get(sourceId) ?? sourceId
          return (
            <div className='leading-tight'>
              <span className='block truncate text-sm font-medium'>{sourceName}</span>
              <span className='block font-mono text-xs text-muted-foreground'>{shortId(sourceId)}</span>
            </div>
          )
        },
      },
      {
        id: 'auth',
        accessorFn: (row) => row.auth_id ?? '-',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Auth' />,
        cell: ({ row }) => {
          const authId = row.original.auth_id
          if (!authId) return <span className='text-muted-foreground'>-</span>
          const authName = authNameById.get(authId) ?? authId
          return <span className='block truncate text-sm font-medium'>{authName}</span>
        },
      },
      {
        id: 'active',
        accessorFn: (row) => (row.active ? 'active' : 'inactive'),
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Active' />,
        cell: ({ row }) => (
          <Badge className={activeBadgeClass(row.original.active)}>
            {row.original.active ? 'Active' : 'Inactive'}
          </Badge>
        ),
      },
      {
        id: 'status',
        accessorFn: (row) => row.runtime?.status ?? 'idle',
        filterFn: arrayFilterFn,
        header: ({ column }) => <DataTableColumnHeader column={column} title='Status' />,
        cell: ({ row }) => (
          <Badge className={statusBadgeClass(row.original.runtime?.status)}>
            {formatStatusLabel(row.original.runtime?.status)}
          </Badge>
        ),
      },
      {
        accessorKey: 'topic',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Topic' />,
        cell: ({ row }) => {
          const topicLabel = displayJobTopic(row.original)
          return (
            <span className='block max-w-[24rem] truncate' title={topicLabel}>
              {topicLabel}
            </span>
          )
        },
      },
      {
        accessorKey: 'last_run_status',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Last Run' />,
        cell: ({ row }) => formatStatusLabel(row.original.last_run_status ?? 'never'),
      },
      {
        accessorKey: 'successful_runs',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Successful Runs' />,
        cell: ({ row }) => Number(row.original.successful_runs ?? 0),
      },
      {
        accessorKey: 'threads_found_total',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Threads Found' />,
        cell: ({ row }) => Number(row.original.threads_found_total ?? 0),
      },
      {
        accessorKey: 'threads_replied_total',
        header: ({ column }) => <DataTableColumnHeader column={column} title='Threads Replied' />,
        cell: ({ row }) => Number(row.original.threads_replied_total ?? 0),
      },
      {
        id: ACTIONS_COLUMN_ID,
        enableSorting: false,
        enableHiding: false,
        cell: ({ row }) => {
          const job = row.original
          return (
            <div className='flex justify-end'>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant='ghost'
                    size='icon'
                    className='size-8'
                    onClick={(event) => event.stopPropagation()}
                  >
                    <MoreHorizontal className='size-4' />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' onClick={(event) => event.stopPropagation()}>
                  <DropdownMenuItem className='gap-2' onSelect={() => openJobSheet(job)}>
                    <Eye className='size-4' />
                    View details
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2' onSelect={() => openDuplicateJobDialog(job)}>
                    <Copy className='size-4' />
                    Duplicate
                  </DropdownMenuItem>
                  {job.active ? (
                    <DropdownMenuItem className='gap-2' onSelect={() => void activationAction(job, 'deactivate')}>
                      <PowerOff className='size-4' />
                      Deactivate
                    </DropdownMenuItem>
                  ) : (
                    <DropdownMenuItem className='gap-2' onSelect={() => void activationAction(job, 'activate')}>
                      <Power className='size-4' />
                      Activate
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem className='gap-2' onSelect={() => void runAction(job, 'start')}>
                    <Play className='size-4' />
                    Start run
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2' onSelect={() => void runAction(job, 'stop')}>
                    <Square className='size-4' />
                    Stop run
                  </DropdownMenuItem>
                  <DropdownMenuItem className='gap-2 text-destructive' onSelect={() => requestDeleteJob(job)}>
                    <Trash2 className='size-4' />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )
        },
      },
    ],
    [activationAction, authNameById, openDuplicateJobDialog, openJobSheet, requestDeleteJob, runAction, sourceSearchNameById]
  )

  const orderedColumnIds = useMemo(
    () =>
      columns
        .map((column) => {
          if (typeof column.id === 'string' && column.id.length > 0) return column.id
          if ('accessorKey' in column && typeof column.accessorKey === 'string' && column.accessorKey.length > 0) {
            return column.accessorKey
          }
          return ''
        })
        .filter((value): value is string => value.length > 0),
    [columns]
  )

  useEffect(() => {
    setColumnOrder((previous) => {
      if (previous.length === 0) return orderedColumnIds
      const previousSet = new Set(previous)
      const next = previous.filter((id) => orderedColumnIds.includes(id))
      orderedColumnIds.forEach((id) => {
        if (!previousSet.has(id)) next.push(id)
      })
      return next
    })
  }, [orderedColumnIds])

  const moveColumn = useCallback((fromId: string, toId: string): void => {
    if (
      fromId === toId ||
      fromId === ACTIONS_COLUMN_ID ||
      toId === ACTIONS_COLUMN_ID ||
      fromId === SELECT_COLUMN_ID ||
      toId === SELECT_COLUMN_ID
    ) return
    setColumnOrder((previous) => {
      const next = previous.length > 0 ? [...previous] : [...orderedColumnIds]
      const fromIndex = next.indexOf(fromId)
      const toIndex = next.indexOf(toId)
      if (fromIndex < 0 || toIndex < 0) return next
      const [moved] = next.splice(fromIndex, 1)
      next.splice(toIndex, 0, moved)
      return next
    })
  }, [orderedColumnIds])

  const table = useReactTable({
    data: jobs,
    columns,
    state: {
      sorting,
      columnFilters,
      globalFilter,
      columnVisibility,
      columnOrder,
      rowSelection,
    },
    enableRowSelection: true,
    enableGlobalFilter: true,
    globalFilterFn: (row, _columnId, filterValue) => {
      const query = String(filterValue ?? '').trim().toLowerCase()
      if (!query) return true
      const job = row.original
      return (
        job.id.toLowerCase().includes(query) ||
        job.name.toLowerCase().includes(query) ||
        job.job_type.toLowerCase().includes(query) ||
        job.social.toLowerCase().includes(query) ||
        (job.auth_id
          ? ((authNameById.get(job.auth_id) ?? job.auth_id).toLowerCase().includes(query))
          : false) ||
        (job.source_job_id
          ? (sourceSearchNameById.get(job.source_job_id) ?? job.source_job_id).toLowerCase().includes(query)
          : false) ||
        displayJobTopic(job).toLowerCase().includes(query) ||
        (job.active ? 'active' : 'inactive').includes(query) ||
        (job.runtime?.status ?? 'idle').toLowerCase().includes(query) ||
        (job.last_run_status ?? 'never').toLowerCase().includes(query)
      )
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onGlobalFilterChange: setGlobalFilter,
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
    onRowSelectionChange: setRowSelection,
    getRowId: (row) => row.id,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getFacetedRowModel: getFacetedRowModel(),
    getFacetedUniqueValues: getFacetedUniqueValues(),
    initialState: {
      pagination: { pageSize: 10 },
    },
  })

  const selectedStartKey = selectedJob ? `${selectedJob.id}:start` : ''
  const selectedStopKey = selectedJob ? `${selectedJob.id}:stop` : ''
  const selectedActivateKey = selectedJob ? `${selectedJob.id}:activate` : ''
  const selectedDeactivateKey = selectedJob ? `${selectedJob.id}:deactivate` : ''
  const selectedDeleteKey = selectedJob ? `${selectedJob.id}:delete` : ''
  const selectedLogs = selectedJob?.runtime?.logs ?? []
  const filteredSelectedLogs = useMemo(() => {
    if (jobActivityFilter === 'all') return selectedLogs
    if (jobActivityFilter === 'errors') {
      return selectedLogs.filter((entry) => /(error|fail|exception)/i.test(entry.message))
    }
    return selectedLogs.filter((entry) => /(warn|retry|slow)/i.test(entry.message))
  }, [jobActivityFilter, selectedLogs])
  const SelectedJobTypeIcon = selectedJob ? jobTypeIcon(selectedJob.job_type) : null

  const renderJobFormFields = (prefix: string) => {
    const availableAuthOptions = authOptions.filter((auth) => auth.social === form.social)
    const availableSearchQueryOptions = searchQueryOptions.filter((item) => {
      if (item.status !== 'saved') return false
      if (item.social !== form.social) return false
      return true
    })
    const isLinkedinSearch = form.job_type === 'search' && form.social === 'linkedin'
    const showRedditSearchFilters = form.job_type === 'search' && form.social === 'reddit'
    const showLinkedinSearchFilters = isLinkedinSearch
    return (
      <>
      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-type`}>Job type</Label>
        <Select
          value={form.job_type}
          onValueChange={(value) => {
            const nextType = value as JobType
            const sourceName =
              nextType === 'reply' && form.source_job_id
                ? sourceSearchNameById.get(form.source_job_id)
                : undefined
            const sourceAuthId =
              nextType === 'reply' && form.source_job_id
                ? sourceSearchAuthById.get(form.source_job_id) ?? null
                : null
            setForm((prev) => ({
              ...prev,
              job_type: nextType,
              source_job_id: nextType === 'reply' ? prev.source_job_id : null,
              topic: nextType === 'reply' ? '' : prev.topic,
              search_plan: undefined,
              search_query_id: nextType === 'search' ? prev.search_query_id : null,
              search_query_version: nextType === 'search' ? prev.search_query_version : null,
              personas: nextType === 'reply' ? ((prev.personas ?? []).length > 0 ? [prev.personas![0]] : []) : [],
              auth_id: nextType === 'reply' ? sourceAuthId : prev.auth_id,
              name: nextType === 'reply' ? extractReplyPrefix(prev.name, sourceName) : prev.name,
            }))
          }}
        >
          <SelectTrigger id={`${prefix}-job-type`} className='w-full'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='search'>
              <span className='inline-flex items-center gap-2'>
                <Search className='size-3.5' />
                Search
              </span>
            </SelectItem>
            <SelectItem value='reply'>
              <span className='inline-flex items-center gap-2'>
                <MessageSquareReply className='size-3.5' />
                Reply
              </span>
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-social`}>Social</Label>
        <Select
          value={form.social}
          onValueChange={(value) => {
            const nextSocial = value as SocialPlatform
            setForm((prev) => {
              const nextAuth = authOptions.find((auth) => auth.id === prev.auth_id)
              const compatibleAuthId = nextAuth && nextAuth.social === nextSocial ? prev.auth_id : null
              return {
                ...prev,
                social: nextSocial,
                auth_id: compatibleAuthId,
                search_plan: undefined,
                search_query_id: null,
                search_query_version: null,
              }
            })
          }}
          disabled={form.job_type === 'reply' && Boolean(form.source_job_id)}
        >
          <SelectTrigger id={`${prefix}-job-social`} className='w-full'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='reddit'>
              <span className='inline-flex items-center gap-2'>
                <FontAwesomeIcon icon={socialIcon('reddit')} className='size-3.5' />
                Reddit
              </span>
            </SelectItem>
            <SelectItem value='linkedin'>
              <span className='inline-flex items-center gap-2'>
                <FontAwesomeIcon icon={socialIcon('linkedin')} className='size-3.5' />
                LinkedIn
              </span>
            </SelectItem>
          </SelectContent>
        </Select>
        {form.job_type === 'reply' && form.source_job_id ? (
          <p className='text-xs text-muted-foreground'>
            Reply jobs inherit social from their source Search job.
          </p>
        ) : null}
      </div>

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-auth`}>Auth profile</Label>
        <Select
          value={form.auth_id ?? undefined}
          onValueChange={(value) =>
            setForm((prev) => ({
              ...prev,
              auth_id: value,
              search_plan: prev.job_type === 'search' ? undefined : prev.search_plan,
              search_query_id: prev.job_type === 'search' ? null : prev.search_query_id,
              search_query_version: prev.job_type === 'search' ? null : prev.search_query_version,
            }))
          }
          disabled={form.job_type === 'reply' && Boolean(form.source_job_id)}
        >
          <SelectTrigger id={`${prefix}-job-auth`} className='w-full'>
            <SelectValue placeholder='Select an auth profile' />
          </SelectTrigger>
          <SelectContent>
            {availableAuthOptions.map((auth) => (
              <SelectItem key={auth.id} value={auth.id}>
                {authDisplayName(auth)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {form.job_type === 'reply' && form.source_job_id ? (
          <p className='text-xs text-muted-foreground'>
            Reply jobs inherit auth from their source Search job.
          </p>
        ) : null}
      </div>

      {form.job_type === 'reply' ? (
        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-source-job-id`}>Source search job</Label>
          <Select
            value={form.source_job_id ?? undefined}
            onValueChange={(value) =>
              setForm((prev) => ({
                ...prev,
                source_job_id: value,
                social: sourceSearchSocialById.get(value) ?? prev.social,
                auth_id: sourceSearchAuthById.get(value) ?? prev.auth_id ?? null,
                name: extractReplyPrefix(prev.name, sourceSearchNameById.get(value)),
              }))
            }
          >
            <SelectTrigger id={`${prefix}-source-job-id`} className='w-full'>
              <SelectValue placeholder='Select a Search job' />
            </SelectTrigger>
            <SelectContent>
              {searchingJobOptions.map((job) => (
                <SelectItem key={job.id} value={job.id}>
                  {job.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className='text-xs text-muted-foreground'>
            Reply jobs read threads from this linked Search job.
          </p>
        </div>
      ) : null}

      <div className='space-y-2'>
        <Label htmlFor={`${prefix}-job-name`}>Name</Label>
        {form.job_type === 'reply' ? (
          <div className='flex items-center gap-2'>
            <Input
              id={`${prefix}-job-name`}
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              placeholder='campaign'
              className='w-full sm:max-w-[12rem]'
              required
            />
            <Input
              id={`${prefix}-job-name-suffix`}
              value={buildReplySuffix(sourceSearchNameById.get(form.source_job_id ?? ''))}
              readOnly
              aria-label='Name suffix'
              className='w-full sm:max-w-[16rem] bg-muted/40 text-muted-foreground'
            />
          </div>
        ) : (
          <Input
            id={`${prefix}-job-name`}
            value={form.name}
            onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
            placeholder='growth'
            required
          />
        )}
        {form.job_type === 'reply' ? (
          <p className='text-xs text-muted-foreground'>
            Final name: {buildReplyJobName(form.name, sourceSearchNameById.get(form.source_job_id ?? ''))}
          </p>
        ) : null}
      </div>

      {form.job_type === 'search' ? (
        <>
          <div className='space-y-2'>
            <Label htmlFor={`${prefix}-search-query-id`}>Search query</Label>
            <Select
              value={form.search_query_id ?? undefined}
              onValueChange={(value) => {
                const selected = searchPathById.get(value)
                setForm((prev) => ({
                  ...prev,
                  search_query_id: value,
                  search_query_version: selected?.version ?? null,
                  social: selected?.social ?? prev.social,
                  topic: selected?.topic ?? prev.topic,
                  search_plan: selected?.plan ?? prev.search_plan,
                }))
              }}
            >
              <SelectTrigger id={`${prefix}-search-query-id`} className='w-full'>
                <SelectValue placeholder='Select a saved search query' />
              </SelectTrigger>
              <SelectContent>
                {availableSearchQueryOptions.map((item) => (
                  <SelectItem key={item.id} value={item.id}>
                    {item.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className='text-xs text-muted-foreground'>
              Pick a saved Search Query. Create or edit drafts from the Search Queries page.
            </p>
          </div>

          <div className={
            showRedditSearchFilters
              ? 'grid gap-3 sm:grid-cols-3'
              : showLinkedinSearchFilters
                ? 'grid gap-3 sm:grid-cols-7'
                : 'grid gap-3 sm:grid-cols-1'
          }>
            {showRedditSearchFilters ? (
              <>
                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-job-time-filter`}>Time filter</Label>
                  <Select
                    value={form.time_filter}
                    onValueChange={(value) => setForm((prev) => ({ ...prev, time_filter: value }))}
                  >
                    <SelectTrigger id={`${prefix}-job-time-filter`} className='w-full'>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value='hour'>hour</SelectItem>
                      <SelectItem value='day'>day</SelectItem>
                      <SelectItem value='week'>week</SelectItem>
                      <SelectItem value='month'>month</SelectItem>
                      <SelectItem value='year'>year</SelectItem>
                      <SelectItem value='all'>all</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-subreddit-limit`}>Subreddit limit</Label>
                  <Input
                    id={`${prefix}-subreddit-limit`}
                    type='number'
                    min={1}
                    value={form.subreddit_limit}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        subreddit_limit: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>
              </>
            ) : null}

            {showLinkedinSearchFilters ? (
              <>
                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-linkedin-time-filter`}>Time filter</Label>
                  <Select
                    value={form.time_filter}
                    onValueChange={(value) => setForm((prev) => ({ ...prev, time_filter: value }))}
                  >
                    <SelectTrigger id={`${prefix}-linkedin-time-filter`} className='w-full'>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value='hour'>hour</SelectItem>
                      <SelectItem value='day'>day</SelectItem>
                      <SelectItem value='week'>week</SelectItem>
                      <SelectItem value='month'>month</SelectItem>
                      <SelectItem value='year'>year</SelectItem>
                      <SelectItem value='all'>all</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-linkedin-sort-by`}>Sort by</Label>
                  <Select
                    value={form.sort_by}
                    onValueChange={(value) => setForm((prev) => ({ ...prev, sort_by: value }))}
                  >
                    <SelectTrigger id={`${prefix}-linkedin-sort-by`} className='w-full'>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value='relevance'>relevance</SelectItem>
                      <SelectItem value='recency'>recency</SelectItem>
                      <SelectItem value='engagement'>engagement</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-scroll-rounds`}>Scroll rounds</Label>
                  <Input
                    id={`${prefix}-scroll-rounds`}
                    type='number'
                    min={1}
                    value={form.scroll_rounds}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        scroll_rounds: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-selector-wait-timeout-seconds`}>Selector wait (s)</Label>
                  <Input
                    id={`${prefix}-selector-wait-timeout-seconds`}
                    type='number'
                    min={1}
                    value={form.selector_wait_timeout_seconds}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        selector_wait_timeout_seconds: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-detail-wait-timeout-seconds`}>Detail wait (s)</Label>
                  <Input
                    id={`${prefix}-detail-wait-timeout-seconds`}
                    type='number'
                    min={1}
                    value={form.detail_wait_timeout_seconds}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        detail_wait_timeout_seconds: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-max-comment-expand-clicks`}>Expand comments clicks</Label>
                  <Input
                    id={`${prefix}-max-comment-expand-clicks`}
                    type='number'
                    min={1}
                    value={form.max_comment_expand_clicks}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        max_comment_expand_clicks: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>

                <div className='space-y-2'>
                  <Label htmlFor={`${prefix}-detail-enrich-top-k`}>Detail enrich posts</Label>
                  <Input
                    id={`${prefix}-detail-enrich-top-k`}
                    type='number'
                    min={1}
                    value={form.detail_enrich_top_k}
                    onChange={(event) =>
                      setForm((prev) => ({
                        ...prev,
                        detail_enrich_top_k: Number(event.target.value || 1),
                      }))
                    }
                  />
                </div>
              </>
            ) : null}

            <div className='space-y-2'>
              <Label htmlFor={`${prefix}-threads-limit`}>{isLinkedinSearch ? 'Posts limit' : 'Threads limit'}</Label>
              <Input
                id={`${prefix}-threads-limit`}
                type='number'
                min={1}
                value={form.threads_limit}
                onChange={(event) =>
                  setForm((prev) => ({
                    ...prev,
                    threads_limit: Number(event.target.value || 1),
                  }))
                }
              />
            </div>
          </div>
        </>
      ) : (
        <>
          <div className='space-y-2'>
            <Label htmlFor={`${prefix}-reply-persona`}>Persona</Label>
            {personaOptions.length > 0 ? (
              <Select
                value={form.personas?.[0] ?? undefined}
                onValueChange={(value) =>
                  setForm((prev) => ({
                    ...prev,
                    personas: [value],
                  }))
                }
              >
                <SelectTrigger id={`${prefix}-reply-persona`} className='w-full'>
                  <SelectValue placeholder='Select a persona' />
                </SelectTrigger>
                <SelectContent>
                  {personaOptions.map((persona) => (
                    <SelectItem key={persona.name} value={persona.name}>
                      {persona.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : (
              <p className='text-sm text-muted-foreground'>
                No personas found. Open the Personas page to add one.
              </p>
            )}
          </div>

          <div className='space-y-2'>
            <Label htmlFor={`${prefix}-min-similarity-score`}>Minimum similarity score</Label>
            <Input
              id={`${prefix}-min-similarity-score`}
              type='number'
              min={0}
              max={1}
              step='0.01'
              value={form.min_similarity_score}
              onChange={(event) =>
                setForm((prev) => {
                  const numeric = Number.parseFloat(event.target.value)
                  const bounded = Number.isFinite(numeric) ? Math.min(1, Math.max(0, numeric)) : 0
                  return {
                    ...prev,
                    min_similarity_score: bounded,
                  }
                })
              }
            />
          </div>
        </>
      )}

      <div className='grid gap-3 sm:grid-cols-2'>
        {form.job_type === 'reply' ? (
          <div className='space-y-2'>
            <Label htmlFor={`${prefix}-replies-per-iteration`}>Replies per iteration</Label>
            <Input
              id={`${prefix}-replies-per-iteration`}
              type='number'
              min={1}
              value={form.replies_per_iteration}
              onChange={(event) =>
                setForm((prev) => ({
                  ...prev,
                  replies_per_iteration: Number(event.target.value || 1),
                }))
              }
            />
          </div>
        ) : null}

        <div className='space-y-2'>
          <Label htmlFor={`${prefix}-max-runtime-minutes`}>Run frequency</Label>
          <Select
            value={String(normalizeRunFrequencyMinutes(form.max_runtime_minutes))}
            onValueChange={(value) =>
              setForm((prev) => ({
                ...prev,
                max_runtime_minutes: Number.parseInt(value, 10),
              }))
            }
          >
            <SelectTrigger id={`${prefix}-max-runtime-minutes`} className='w-full'>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value='60'>Every 1 hour</SelectItem>
              <SelectItem value='360'>Every 6 hours</SelectItem>
              <SelectItem value='720'>Every 12 hours</SelectItem>
              <SelectItem value='1440'>Every 24 hours</SelectItem>
            </SelectContent>
          </Select>
          <p className='text-xs text-muted-foreground'>How often this job runs.</p>
        </div>
      </div>
    </>
    )
  }

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <div className='inline-flex h-8 items-center overflow-hidden rounded-md border'>
            <Button
              variant='ghost'
              size='icon'
              className='h-full w-8 rounded-none border-r'
              onClick={() => void load()}
              disabled={loading}
              aria-label='Manual refresh jobs'
              title='Manual refresh jobs'
            >
              <RefreshCw className={cn('size-4', autoRefreshSeconds > 0 && 'animate-spin [animation-duration:2s]')} />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant='ghost' size='sm' aria-label='Auto-refresh jobs' className='h-full rounded-none gap-2 px-3'>
                  <ChevronDown className='size-4' />
                  <span>{autoRefreshLabel}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align='end'>
                {AUTO_REFRESH_OPTIONS.map((option) => (
                  <DropdownMenuItem key={option.value} onSelect={() => setAutoRefreshSeconds(option.value)}>
                    <span className='mr-3 inline-flex w-4 justify-center text-foreground'>
                      {autoRefreshSeconds === option.value ? '●' : ''}
                    </span>
                    {option.menuLabel}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <Button size='sm' onClick={openCreateDialog}>
            <Plus className='size-4' />
            Create job
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Jobs</h2>
        </div>

        <DataTableToolbar
          table={table}
          searchPlaceholder='Search by id, name, topic or status...'
          filters={[
            ...(jobTypeFilterOptions.length > 0
              ? [{ columnId: 'job_type', title: 'Type', options: jobTypeFilterOptions }]
              : []),
            ...(activeFilterOptions.length > 0
              ? [{ columnId: 'active', title: 'Active', options: activeFilterOptions }]
              : []),
            ...(statusFilterOptions.length > 0
              ? [{ columnId: 'status', title: 'Status', options: statusFilterOptions }]
              : []),
          ]}
        />

        <div className='space-y-4'>
          <div className='rounded-md border'>
            <Table>
              <TableHeader>
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => (
                      <TableHead
                        key={header.id}
                        className={cn(
                          header.column.id === draggingColumnId && 'opacity-50',
                          header.column.id !== ACTIONS_COLUMN_ID && header.column.id !== SELECT_COLUMN_ID && 'cursor-move',
                          header.column.id === ACTIONS_COLUMN_ID &&
                            'sticky right-0 z-20 bg-background text-right'
                        )}
                        draggable={header.column.id !== ACTIONS_COLUMN_ID && header.column.id !== SELECT_COLUMN_ID}
                        onDragStart={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
                          setDraggingColumnId(header.column.id)
                          event.dataTransfer.effectAllowed = 'move'
                          event.dataTransfer.setData('text/plain', header.column.id)
                        }}
                        onDragOver={(event) => {
                          if (!draggingColumnId || header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
                          if (draggingColumnId === header.column.id) return
                          event.preventDefault()
                          event.dataTransfer.dropEffect = 'move'
                        }}
                        onDrop={(event) => {
                          if (header.column.id === ACTIONS_COLUMN_ID || header.column.id === SELECT_COLUMN_ID) return
                          event.preventDefault()
                          const source = draggingColumnId ?? event.dataTransfer.getData('text/plain')
                          if (!source) return
                          moveColumn(source, header.column.id)
                          setDraggingColumnId(null)
                        }}
                        onDragEnd={() => setDraggingColumnId(null)}
                      >
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </TableHead>
                    ))}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows.length ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow
                      key={row.id}
                      className='cursor-pointer'
                      data-state={row.getIsSelected() && 'selected'}
                      onClick={() => openJobSheet(row.original)}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell
                          key={cell.id}
                          className={cn(
                            cell.column.id === ACTIONS_COLUMN_ID &&
                              'sticky right-0 z-10 bg-background'
                          )}
                        >
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={table.getVisibleLeafColumns().length} className='h-24 text-center text-muted-foreground'>
                      No jobs configured.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>

          <DataTablePagination table={table} />
        </div>
      </Main>

      <SelectionActionBar
        selectedCount={table.getSelectedRowModel().rows.length}
        entityLabel='job'
        onClear={() => setRowSelection({})}
        onDelete={() => requestDeleteSelectedJobs()}
        deleteLabel='Delete selected'
        deleteDisabled={busyKey === 'bulk-delete'}
      />

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent className='sm:max-w-2xl'>
          <ModalDialogHeader>
            <ModalDialogTitle>Create job</ModalDialogTitle>
            <DialogDescription>
              Create a Search job (search + rank) or a Reply job (reply from a linked Search artifact).
            </DialogDescription>
          </ModalDialogHeader>

          <div className='max-h-[70vh] space-y-4 overflow-y-auto pr-1'>
            {renderJobFormFields('dialog')}
          </div>

          <DialogFooter>
            <Button
              type='button'
              variant='outline'
              onClick={() => setCreateDialogOpen(false)}
            >
              Cancel
            </Button>
            <Button type='button' onClick={() => void createJob()} disabled={saving}>
              {saving ? 'Saving...' : 'Create job'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={pendingDeleteJob !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteJob(null)
        }}
        title='Delete job'
        desc={
          pendingDeleteJob
            ? pendingDeleteJobRelatedReplies.length > 0
              ? (
                  <>
                    <p>This action will delete "{pendingDeleteJob.name}" and related reply jobs.</p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related replies: {pendingDeleteJobRelatedReplies.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `This action will permanently delete "${pendingDeleteJob.name}".`
            : 'This action will permanently delete this job.'
        }
        confirmText='Delete'
        destructive
        isLoading={pendingDeleteJob !== null && busyKey === `${pendingDeleteJob.id}:delete`}
        handleConfirm={() => void confirmDeleteJob()}
      />

      <ConfirmDialog
        open={pendingDeleteJobs.length > 0}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteJobs([])
        }}
        title='Delete selected jobs'
        desc={
          pendingDeleteJobs.length > 0
            ? pendingDeleteJobsRelatedReplies.length > 0
              ? (
                  <>
                    <p>
                      This will delete {pendingDeleteJobs.length} selected jobs and {pendingDeleteJobsRelatedReplies.length} related reply jobs.
                    </p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related replies: {pendingDeleteJobsRelatedReplies.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `This action will permanently delete ${pendingDeleteJobs.length} selected job${pendingDeleteJobs.length === 1 ? '' : 's'}.`
            : ''
        }
        confirmText='Delete'
        destructive
        isLoading={busyKey === 'bulk-delete'}
        handleConfirm={() => void confirmDeleteSelectedJobs()}
      />

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='sm:max-w-xl p-0'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedJob ? `Job · ${selectedJob.name}` : 'Job details'}</SheetTitle>
              <SheetDescription>
                Runtime controls and job details.
              </SheetDescription>
              {selectedJob ? (
                <div className='mt-2 flex flex-wrap items-center gap-1.5'>
                  <Badge className='bg-muted text-muted-foreground'>ID {shortId(selectedJob.id)}</Badge>
                  <Badge className='bg-muted text-muted-foreground'>
                    <span className='inline-flex items-center gap-1.5'>
                      {SelectedJobTypeIcon ? <SelectedJobTypeIcon className='size-3.5' /> : null}
                      Type {formatJobType(selectedJob.job_type)}
                    </span>
                  </Badge>
                  <Badge className='bg-muted text-muted-foreground'>
                    <span className='inline-flex items-center gap-1.5'>
                      <FontAwesomeIcon icon={socialIcon(selectedJob.social)} className='size-3.5' />
                      {formatSocial(selectedJob.social)}
                    </span>
                  </Badge>
                  <Badge className={activeBadgeClass(selectedJob.active)}>
                    {selectedJob.active ? 'Active' : 'Inactive'}
                  </Badge>
                  <Badge className={statusBadgeClass(selectedJob.runtime?.status)}>
                    {formatStatusLabel(selectedJob.runtime?.status)}
                  </Badge>
                </div>
              ) : null}
            </SheetHeader>

            <div className='sticky top-0 z-10 mt-3 flex items-center gap-2 border-y bg-background px-4 py-3'>
              {selectedJob ? (
                <>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() =>
                      void activationAction(selectedJob, selectedJob.active ? 'deactivate' : 'activate')
                    }
                    disabled={
                      busyKey === selectedActivateKey ||
                      busyKey === selectedDeactivateKey
                    }
                  >
                    {selectedJob.active ? 'Deactivate' : 'Activate'}
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => void runAction(selectedJob, 'start')}
                    disabled={busyKey === selectedStartKey}
                  >
                    <Play className='size-4' />
                    Start run
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => void runAction(selectedJob, 'stop')}
                    disabled={busyKey === selectedStopKey}
                  >
                    <Square className='size-4' />
                    Stop run
                  </Button>
                  <Button
                    size='sm'
                    variant='outline'
                    onClick={() => requestDeleteJob(selectedJob)}
                    disabled={busyKey === selectedDeleteKey}
                  >
                    <Trash2 className='size-4' />
                    Delete
                  </Button>
                </>
              ) : null}
            </div>

            <Tabs
              value={sheetTab}
              onValueChange={(value) => setSheetTab(value as 'overview' | 'activity')}
              className='flex-1 overflow-hidden px-4 py-4'
            >
              <TabsList className='w-full justify-start'>
                <TabsTrigger value='overview'>Overview</TabsTrigger>
                <TabsTrigger value='activity'>Activity</TabsTrigger>
              </TabsList>

              <TabsContent value='overview' className='min-h-0 overflow-y-auto pb-4'>
                {selectedJob ? (
                  <div className='space-y-4 text-sm'>
                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Overview
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        <p><span className='font-medium'>Name:</span> {selectedJob.name}</p>
                        <p><span className='font-medium'>ID:</span> <span className='font-mono'>{selectedJob.id}</span></p>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        State
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        <p><span className='font-medium'>Active:</span> {selectedJob.active ? 'Yes' : 'No'}</p>
                        <p><span className='font-medium'>Current status:</span> {formatStatusLabel(selectedJob.runtime?.status)}</p>
                        <p><span className='font-medium'>Last run status:</span> {formatStatusLabel(selectedJob.last_run_status ?? 'never')}</p>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Metrics
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        <p><span className='font-medium'>Successful runs:</span> {selectedJob.successful_runs ?? 0}</p>
                        <p><span className='font-medium'>Threads found:</span> {selectedJob.threads_found_total ?? 0}</p>
                        <p><span className='font-medium'>Threads replied:</span> {selectedJob.threads_replied_total ?? 0}</p>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Scheduling
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        <p><span className='font-medium'>Run frequency:</span> {formatRunFrequency(selectedJob.max_runtime_minutes)}</p>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        Context
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        <p>
                          <span className='font-medium'>Auth:</span>{' '}
                          {selectedJob.auth_id ? (authNameById.get(selectedJob.auth_id) ?? selectedJob.auth_id) : '-'}
                        </p>
                        <p>
                          <span className='font-medium'>Source search job:</span>{' '}
                          {selectedJob.source_job_id ? (sourceSearchNameById.get(selectedJob.source_job_id) ?? selectedJob.source_job_id) : '-'}
                        </p>
                        <p><span className='font-medium'>Topic:</span> {selectedJob.job_type === 'reply' ? 'N/A' : (selectedJob.topic ?? '-')}</p>
                        <p>
                          <span className='font-medium'>Search query:</span>{' '}
                          {selectedJob.search_query_id
                            ? selectedJob.search_query_id
                            : '-'}
                        </p>
                      </div>
                    </section>

                    <section className='rounded-lg border bg-muted/20 p-3'>
                      <h3 className='text-xs font-semibold uppercase tracking-wide text-muted-foreground'>
                        {selectedJob.job_type === 'search' ? 'Search configuration' : 'Reply configuration'}
                      </h3>
                      <div className='mt-2 grid gap-2'>
                        {selectedJob.job_type === 'search' && selectedJob.social === 'reddit' ? (
                          <>
                            <p><span className='font-medium'>Time filter:</span> {selectedJob.time_filter}</p>
                            <p><span className='font-medium'>Subreddit limit:</span> {selectedJob.subreddit_limit}</p>
                            <p><span className='font-medium'>Locked subreddits:</span> {(selectedJob.search_plan?.subreddits ?? []).join(', ') || '-'}</p>
                            <p><span className='font-medium'>Locked queries:</span> {(selectedJob.search_plan?.queries ?? []).join(' | ') || '-'}</p>
                          </>
                        ) : null}
                        {selectedJob.job_type === 'search' && selectedJob.social === 'linkedin' ? (
                          <>
                            <p><span className='font-medium'>Time filter:</span> {selectedJob.time_filter}</p>
                            <p><span className='font-medium'>Sort by:</span> {selectedJob.sort_by}</p>
                            <p><span className='font-medium'>Selector wait (s):</span> {selectedJob.selector_wait_timeout_seconds}</p>
                            <p><span className='font-medium'>Detail wait (s):</span> {selectedJob.detail_wait_timeout_seconds}</p>
                            <p><span className='font-medium'>Expand comments clicks:</span> {selectedJob.max_comment_expand_clicks}</p>
                            <p><span className='font-medium'>Detail enrich posts:</span> {selectedJob.detail_enrich_top_k}</p>
                            <p><span className='font-medium'>Locked query:</span> {selectedJob.search_plan?.query || '-'}</p>
                          </>
                        ) : null}
                        <p>
                          <span className='font-medium'>{selectedJob.social === 'linkedin' ? 'Posts limit' : 'Threads limit'}:</span>{' '}
                          {selectedJob.job_type === 'search' ? selectedJob.threads_limit : 'N/A'}
                        </p>
                        <p>
                          <span className='font-medium'>Scroll rounds:</span>{' '}
                          {selectedJob.job_type === 'search' && selectedJob.social === 'linkedin' ? selectedJob.scroll_rounds : 'N/A'}
                        </p>
                        {selectedJob.job_type === 'reply' ? (
                          <p><span className='font-medium'>Replies per iteration:</span> {selectedJob.replies_per_iteration}</p>
                        ) : null}
                      </div>
                    </section>
                  </div>
                ) : null}
              </TabsContent>

              <TabsContent value='activity' className='min-h-0 overflow-y-auto pb-4'>
                <div className='space-y-2'>
                  <div className='flex items-center justify-between gap-2'>
                    <Label>Run timeline</Label>
                    <div className='inline-flex items-center gap-1 rounded-md border bg-background p-1'>
                      <Button
                        size='sm'
                        variant={jobActivityFilter === 'all' ? 'default' : 'ghost'}
                        className='h-7 px-2 text-xs'
                        onClick={() => setJobActivityFilter('all')}
                      >
                        All
                      </Button>
                      <Button
                        size='sm'
                        variant={jobActivityFilter === 'errors' ? 'default' : 'ghost'}
                        className='h-7 px-2 text-xs'
                        onClick={() => setJobActivityFilter('errors')}
                      >
                        Show errors only
                      </Button>
                      <Button
                        size='sm'
                        variant={jobActivityFilter === 'warnings' ? 'default' : 'ghost'}
                        className='h-7 px-2 text-xs'
                        onClick={() => setJobActivityFilter('warnings')}
                      >
                        Show warnings only
                      </Button>
                    </div>
                  </div>
                  <div className='max-h-[70vh] overflow-y-auto rounded-lg border bg-muted/20 p-3'>
                    {filteredSelectedLogs.length > 0 ? (
                      <div className='space-y-2'>
                        {filteredSelectedLogs.map((entry, index) => (
                          <p key={`${entry.at}-${index}`} className='font-mono text-xs leading-5 text-muted-foreground'>
                            [{formatWhen(entry.at)}] {entry.message}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <p className='text-sm text-muted-foreground'>No timeline entries for this filter.</p>
                    )}
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
