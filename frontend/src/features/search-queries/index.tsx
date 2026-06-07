import { useCallback, useEffect, useMemo, useState } from 'react'
import { faLinkedinIn, faRedditAlien } from '@fortawesome/free-brands-svg-icons'
import { type IconDefinition } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Lock, Plus, RefreshCw, Search, Sparkles, Trash2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/confirm-dialog'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { SelectionActionBar } from '@/components/selection-action-bar'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
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
import { Textarea } from '@/components/ui/textarea'
import {
  reddropApi,
  type Job,
  type SearchPlan,
  type SearchQuery,
  type SocialPlatform,
} from '@/lib/reddrop-api'
import { toast } from 'sonner'

type SearchQueryForm = {
  name: string
  social: SocialPlatform
  topic: string
  plan: SearchPlan
}

const DEFAULT_LINKEDIN_QUERY_INPUT_COUNT = 5

function buildInitialForm(): SearchQueryForm {
  return {
    name: '',
    social: 'reddit',
    topic: '',
    plan: {
      topics: [],
      subreddits: [],
      queries: [],
      query: '',
    },
  }
}

function parseLines(raw: string): string[] {
  const normalized: string[] = []
  const seen = new Set<string>()
  raw
    .split('\n')
    .map((item) => item.trim())
    .forEach((item) => {
      if (!item) return
      const lowered = item.toLowerCase()
      if (seen.has(lowered)) return
      seen.add(lowered)
      normalized.push(item)
    })
  return normalized
}

function toLines(values: string[] | undefined): string {
  if (!Array.isArray(values) || values.length === 0) return ''
  return values.join('\n')
}

function firstNonEmpty(values: string[]): string {
  return values.find((item) => item.trim())?.trim() ?? ''
}

function linkedinQuerySlots(values: string[] | undefined, fallback: string | undefined): string[] {
  const source = Array.isArray(values) ? [...values] : []
  if (source.length === 0 && fallback?.trim()) {
    source.push(fallback.trim())
  }
  return Array.from({ length: DEFAULT_LINKEDIN_QUERY_INPUT_COUNT }, (_, index) => source[index] ?? '')
}

function socialIcon(platform: SocialPlatform): IconDefinition {
  return platform === 'linkedin' ? faLinkedinIn : faRedditAlien
}

function formatSocial(value: SocialPlatform): string {
  return value === 'linkedin' ? 'LinkedIn' : 'Reddit'
}

export function SearchQueriesPage() {
  const [queries, setQueries] = useState<SearchQuery[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [savingDraft, setSavingDraft] = useState(false)
  const [savingQuery, setSavingQuery] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)
  const [sheetOpen, setSheetOpen] = useState(false)
  const [selectedQuery, setSelectedQuery] = useState<SearchQuery | null>(null)
  const [selectedQueryIds, setSelectedQueryIds] = useState<Record<string, boolean>>({})
  const [searchText, setSearchText] = useState('')
  const [pendingDeleteQuery, setPendingDeleteQuery] = useState<{ query: SearchQuery; relatedJobs: Job[] } | null>(null)
  const [pendingDeleteQueries, setPendingDeleteQueries] = useState<{ queries: SearchQuery[]; relatedJobs: Job[] } | null>(null)
  const [form, setForm] = useState<SearchQueryForm>(buildInitialForm)

  const load = useCallback(async (): Promise<void> => {
    setLoading(true)
    try {
      const [queriesPayload, jobsPayload] = await Promise.all([
        reddropApi.listSearchQueries(),
        reddropApi.listJobs(),
      ])
      setQueries(queriesPayload.search_queries)
      setJobs(jobsPayload.jobs)
      setSelectedQueryIds((prev) => {
        const valid = new Set(queriesPayload.search_queries.map((item) => item.id))
        return Object.fromEntries(Object.entries(prev).filter(([id, selected]) => selected && valid.has(id)))
      })
      setSelectedQuery((prev) => {
        if (!prev) return null
        return queriesPayload.search_queries.find((item) => item.id === prev.id) ?? null
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load search queries'
      toast.error(message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const openCreate = useCallback(() => {
    setForm(buildInitialForm())
    setCreateOpen(true)
  }, [])

  const openSheet = useCallback((query: SearchQuery) => {
    const planQueries =
      query.plan.queries && query.plan.queries.length > 0
        ? [...query.plan.queries]
        : query.plan.query
          ? [query.plan.query]
          : []
    const normalizedQueries =
      query.social === 'linkedin'
        ? linkedinQuerySlots(planQueries, query.plan.query ?? '')
        : planQueries
    setSelectedQuery(query)
    setForm({
      name: query.name,
      social: query.social,
      topic: query.topic,
      plan: {
        topics: [...(query.plan.topics ?? [])],
        subreddits: [...(query.plan.subreddits ?? [])],
        queries: normalizedQueries,
        query: query.plan.query ?? firstNonEmpty(normalizedQueries),
      },
    })
    setSheetOpen(true)
  }, [])

  const updateLinkedinQuery = useCallback((index: number, value: string) => {
    setForm((prev) => {
      const queries =
        prev.plan.queries && prev.plan.queries.length > 0
          ? [...prev.plan.queries]
          : ['']
      queries[index] = value
      return {
        ...prev,
        plan: {
          ...prev.plan,
          queries,
          query: firstNonEmpty(queries),
        },
      }
    })
  }, [])

  const addLinkedinQuery = useCallback(() => {
    setForm((prev) => {
      const queries =
        prev.plan.queries && prev.plan.queries.length > 0
          ? [...prev.plan.queries, '']
          : [...linkedinQuerySlots(prev.plan.queries, prev.plan.query ?? ''), '']
      return {
        ...prev,
        plan: {
          ...prev.plan,
          queries,
          query: firstNonEmpty(queries),
        },
      }
    })
  }, [])

  const removeLinkedinQuery = useCallback((index: number) => {
    setForm((prev) => {
      const source =
        prev.plan.queries && prev.plan.queries.length > 0
          ? [...prev.plan.queries]
          : ['']
      const queries = source.filter((_, itemIndex) => itemIndex !== index)
      const nextQueries = queries.length > 0 ? queries : ['']
      return {
        ...prev,
        plan: {
          ...prev.plan,
          queries: nextQueries,
          query: firstNonEmpty(nextQueries),
        },
      }
    })
  }, [])

  const generateSearchQuery = useCallback(async () => {
    if (!form.topic.trim()) {
      toast.error('Topic is required')
      return
    }
    setGenerating(true)
    try {
      const generated = await reddropApi.generateSearchQuery({
        name: form.name.trim() || undefined,
        social: form.social,
        topic: form.topic.trim(),
      })
      toast.success('Search query generated')
      setCreateOpen(false)
      await load()
      openSheet(generated.search_query)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to generate search query'
      toast.error(message)
    } finally {
      setGenerating(false)
    }
  }, [form, load, openSheet])

  const saveDraft = useCallback(async () => {
    if (!selectedQuery) return
    if (selectedQuery.status !== 'draft') {
      toast.error('Only draft queries can be edited')
      return
    }
    if (!form.name.trim()) {
      toast.error('Name is required')
      return
    }
    if (!form.topic.trim()) {
      toast.error('Topic is required')
      return
    }

    setSavingDraft(true)
    try {
      const updated = await reddropApi.updateSearchQuery(selectedQuery.id, {
        name: form.name.trim(),
        social: form.social,
        topic: form.topic.trim(),
        plan: form.plan,
      })
      toast.success('Search query updated')
      await load()
      openSheet(updated.search_query)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update search query'
      toast.error(message)
    } finally {
      setSavingDraft(false)
    }
  }, [form, load, openSheet, selectedQuery])

  const saveQuery = useCallback(async () => {
    if (!selectedQuery) return
    setSavingQuery(true)
    try {
      const saved = await reddropApi.saveSearchQuery(selectedQuery.id)
      toast.success('Search query saved')
      await load()
      openSheet(saved.search_query)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to save search query'
      toast.error(message)
    } finally {
      setSavingQuery(false)
    }
  }, [load, openSheet, selectedQuery])

  const relatedJobsForQuery = useCallback((queryId: string): Job[] => {
    const normalizedQueryId = queryId.trim()
    if (!normalizedQueryId) return []
    const relatedById = new Map<string, Job>()
    const linkedSearchJobs = jobs.filter(
      (item) => item.job_type === 'search' && (item.search_query_id ?? '').trim() === normalizedQueryId
    )
    const searchIds = new Set(linkedSearchJobs.map((item) => item.id))
    for (const searchJob of linkedSearchJobs) {
      relatedById.set(searchJob.id, searchJob)
    }
    for (const item of jobs) {
      if (item.job_type !== 'reply') continue
      if (!item.source_job_id) continue
      if (searchIds.has(item.source_job_id)) {
        relatedById.set(item.id, item)
      }
    }
    return Array.from(relatedById.values())
  }, [jobs])

  const deleteSearchQuery = useCallback(async (query: SearchQuery, cascade: boolean): Promise<void> => {
    setSavingDraft(true)
    try {
      if (cascade) await reddropApi.deleteSearchQueryCascade(query.id)
      else await reddropApi.deleteSearchQuery(query.id)
      setSelectedQueryIds((prev) => {
        const next = { ...prev }
        delete next[query.id]
        return next
      })
      if (selectedQuery?.id === query.id) {
        setSelectedQuery(null)
        setSheetOpen(false)
      }
      await load()
      toast.success('Search query deleted')
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete search query'
      toast.error(message)
    } finally {
      setSavingDraft(false)
    }
  }, [load, selectedQuery?.id])

  const requestDeleteSearchQuery = useCallback((query: SearchQuery): void => {
    setPendingDeleteQuery({
      query,
      relatedJobs: relatedJobsForQuery(query.id),
    })
  }, [relatedJobsForQuery])

  const requestDeleteSelectedQueries = useCallback((): void => {
    const ids = Object.entries(selectedQueryIds)
      .filter(([, selected]) => Boolean(selected))
      .map(([id]) => id)
    const selected = queries.filter((item) => ids.includes(item.id))
    if (selected.length === 0) return
    const relatedById = new Map<string, Job>()
    for (const query of selected) {
      for (const related of relatedJobsForQuery(query.id)) {
        relatedById.set(related.id, related)
      }
    }
    setPendingDeleteQueries({
      queries: selected,
      relatedJobs: Array.from(relatedById.values()),
    })
  }, [queries, relatedJobsForQuery, selectedQueryIds])

  const deleteSelectedQueries = useCallback(async (): Promise<void> => {
    if (!pendingDeleteQueries) return
    setSavingDraft(true)
    let deletedCount = 0
    let firstError = ''
    for (const query of pendingDeleteQueries.queries) {
      try {
        const related = relatedJobsForQuery(query.id)
        if (related.length > 0) await reddropApi.deleteSearchQueryCascade(query.id)
        else await reddropApi.deleteSearchQuery(query.id)
        deletedCount += 1
      } catch (error) {
        if (!firstError) {
          firstError = error instanceof Error ? error.message : `Failed to delete ${query.name}`
        }
      }
    }
    setPendingDeleteQueries(null)
    setSelectedQueryIds({})
    await load()
    if (deletedCount > 0) {
      toast.success(`Deleted ${deletedCount} search query${deletedCount === 1 ? '' : 's'}`)
    }
    if (firstError) {
      toast.error(firstError)
    }
    setSavingDraft(false)
  }, [load, pendingDeleteQueries, relatedJobsForQuery])

  const selectedCount = Object.values(selectedQueryIds).filter(Boolean).length
  const filteredQueries = useMemo(() => {
    const query = searchText.trim().toLowerCase()
    if (!query) return queries
    return queries.filter((item) => {
      const name = item.name.toLowerCase()
      const topic = item.topic.toLowerCase()
      const social = formatSocial(item.social).toLowerCase()
      const status = item.status.toLowerCase()
      return name.includes(query) || topic.includes(query) || social.includes(query) || status.includes(query)
    })
  }, [queries, searchText])

  return (
    <>
      <Header fixed>
        <div className='ms-auto flex items-center gap-2'>
          <Button size='sm' variant='outline' onClick={() => void load()} disabled={loading}>
            <RefreshCw className='size-4' />
            Refresh
          </Button>
          <Button size='sm' onClick={openCreate}>
            <Plus className='size-4' />
            Create search query
          </Button>
        </div>
      </Header>

      <Main className='flex flex-1 flex-col gap-4 sm:gap-6'>
        <div>
          <h2 className='text-2xl font-bold tracking-tight'>Search Queries</h2>
        </div>

        <div className='flex items-center gap-2'>
          <div className='relative w-full max-w-sm'>
            <Search className='pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground' />
            <Input
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              placeholder='Search by name, topic, social or status...'
              className='pl-8'
            />
          </div>
        </div>

        <div className='rounded-md border'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className='w-[40px]'>
                  <Checkbox
                    aria-label='Select all search queries'
                    checked={
                      filteredQueries.length > 0 && filteredQueries.every((item) => Boolean(selectedQueryIds[item.id]))
                        ? true
                        : filteredQueries.some((item) => Boolean(selectedQueryIds[item.id]))
                          ? 'indeterminate'
                          : false
                    }
                    onCheckedChange={(checked) => {
                      const nextValue = Boolean(checked)
                      if (!nextValue) {
                        setSelectedQueryIds({})
                        return
                      }
                      setSelectedQueryIds(Object.fromEntries(filteredQueries.map((item) => [item.id, true])))
                    }}
                  />
                </TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Social</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Topic</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredQueries.length > 0 ? (
                filteredQueries.map((query) => {
                  const isSelected = Boolean(selectedQueryIds[query.id])
                  return (
                    <TableRow key={query.id} className='cursor-pointer' onClick={() => openSheet(query)}>
                      <TableCell onClick={(event) => event.stopPropagation()}>
                        <Checkbox
                          aria-label={`Select search query ${query.name}`}
                          checked={isSelected}
                          onCheckedChange={(checked) =>
                            setSelectedQueryIds((prev) => ({
                              ...prev,
                              [query.id]: Boolean(checked),
                            }))
                          }
                        />
                      </TableCell>
                      <TableCell className='font-medium'>{query.name}</TableCell>
                      <TableCell>
                        <Badge className='bg-muted text-muted-foreground'>
                          <span className='inline-flex items-center gap-1.5'>
                            <FontAwesomeIcon icon={socialIcon(query.social)} className='size-3.5' />
                            {formatSocial(query.social)}
                          </span>
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge className={query.status === 'saved' ? 'bg-emerald-600 text-white' : 'bg-amber-500 text-black'}>
                          {query.status}
                        </Badge>
                      </TableCell>
                      <TableCell className='max-w-[24rem] truncate' title={query.topic}>{query.topic}</TableCell>
                    </TableRow>
                  )
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={5} className='h-24 text-center text-muted-foreground'>
                    No search queries found.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </Main>

      <SelectionActionBar
        selectedCount={selectedCount}
        entityLabel='search query'
        onClear={() => setSelectedQueryIds({})}
        onDelete={() => requestDeleteSelectedQueries()}
        deleteLabel='Delete selected'
        deleteDisabled={savingDraft}
      />

      <ConfirmDialog
        open={pendingDeleteQuery !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteQuery(null)
        }}
        title='Delete search query'
        desc={
          pendingDeleteQuery
            ? pendingDeleteQuery.relatedJobs.length > 0
              ? (
                  <>
                    <p>
                      This action will delete "{pendingDeleteQuery.query.name}" and related jobs.
                    </p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related jobs: {pendingDeleteQuery.relatedJobs.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `This action will permanently delete "${pendingDeleteQuery.query.name}".`
            : 'This action will permanently delete this search query.'
        }
        confirmText='Delete'
        destructive
        isLoading={savingDraft}
        handleConfirm={() => {
          if (!pendingDeleteQuery) return
          void deleteSearchQuery(pendingDeleteQuery.query, pendingDeleteQuery.relatedJobs.length > 0)
          setPendingDeleteQuery(null)
        }}
      />

      <ConfirmDialog
        open={pendingDeleteQueries !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeleteQueries(null)
        }}
        title='Delete selected search queries'
        desc={
          pendingDeleteQueries
            ? pendingDeleteQueries.relatedJobs.length > 0
              ? (
                  <>
                    <p>
                      This will delete {pendingDeleteQueries.queries.length} selected search queries and related jobs.
                    </p>
                    <p className='mt-2 text-xs text-muted-foreground'>
                      Related jobs: {pendingDeleteQueries.relatedJobs.map((item) => item.name).join(', ')}
                    </p>
                  </>
                )
              : `This action will permanently delete ${pendingDeleteQueries.queries.length} selected search query${pendingDeleteQueries.queries.length === 1 ? '' : 's'}.`
            : ''
        }
        confirmText='Delete'
        destructive
        isLoading={savingDraft}
        handleConfirm={() => void deleteSelectedQueries()}
      />

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className='sm:max-w-xl'>
          <DialogHeader>
            <DialogTitle>Create search query</DialogTitle>
            <DialogDescription>
              Generate a draft Search Query with LLM assistance, then edit and save it.
            </DialogDescription>
          </DialogHeader>

          <div className='space-y-3'>
            <div className='space-y-2'>
              <Label htmlFor='create-query-name'>Name</Label>
              <Input
                id='create-query-name'
                value={form.name}
                onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
                placeholder='crm-research-query'
              />
            </div>

            <div className='grid gap-3 sm:grid-cols-2'>
              <div className='space-y-2'>
                <Label htmlFor='create-query-social'>Social</Label>
                <Select
                  value={form.social}
                  onValueChange={(value) =>
                    setForm((prev) => ({
                      ...prev,
                      social: value as SocialPlatform,
                    }))
                  }
                >
                  <SelectTrigger id='create-query-social'>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value='reddit'>Reddit</SelectItem>
                    <SelectItem value='linkedin'>LinkedIn</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className='space-y-2'>
              <Label htmlFor='create-query-topic'>Topic</Label>
              <Textarea
                id='create-query-topic'
                rows={4}
                value={form.topic}
                onChange={(event) => setForm((prev) => ({ ...prev, topic: event.target.value }))}
                placeholder='crm sales'
              />
            </div>
          </div>

          <DialogFooter>
            <Button type='button' variant='outline' onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button type='button' onClick={() => void generateSearchQuery()} disabled={generating}>
              <Sparkles className='size-4' />
              {generating ? 'Generating...' : 'Generate search query'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
        <SheetContent side='right' className='p-0 sm:max-w-2xl'>
          <div className='flex h-full flex-col'>
            <SheetHeader className='px-4 pt-4'>
              <SheetTitle>{selectedQuery ? selectedQuery.name : 'Search query details'}</SheetTitle>
              <SheetDescription>
                Review query metadata, adjust draft plan rows, then save.
              </SheetDescription>
            </SheetHeader>

            {selectedQuery ? (
              <>
                <div className='sticky top-0 z-10 mt-3 border-y bg-background px-4 py-3'>
                  <div className='flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between'>
                    <div className='flex min-w-0 flex-wrap items-center gap-1.5'>
                      <Badge className='max-w-[16rem] truncate bg-muted text-muted-foreground' title={selectedQuery.id}>
                        {selectedQuery.id}
                      </Badge>
                      <Badge className={selectedQuery.status === 'saved' ? 'bg-emerald-600 text-white' : 'bg-amber-500 text-black'}>
                        {selectedQuery.status}
                      </Badge>
                    </div>
                    <div className='flex w-full flex-wrap items-center justify-end gap-2 sm:w-auto'>
                      {selectedQuery.status === 'draft' ? (
                        <>
                          <Button
                            type='button'
                            size='sm'
                            variant='outline'
                            onClick={() => void saveDraft()}
                            disabled={savingDraft}
                          >
                            {savingDraft ? 'Saving...' : 'Save draft'}
                          </Button>
                          <Button type='button' size='sm' onClick={() => void saveQuery()} disabled={savingQuery}>
                            <Lock className='size-4' />
                            {savingQuery ? 'Saving...' : 'Save query'}
                          </Button>
                        </>
                      ) : null}
                      <Button
                        type='button'
                        size='sm'
                        variant='outline'
                        onClick={() => requestDeleteSearchQuery(selectedQuery)}
                        disabled={savingDraft}
                      >
                        <Trash2 className='size-4' />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>

                <div className='flex-1 overflow-y-auto px-4 py-4'>
                  <section className='border-b pb-4'>
                    <div className='flex items-center justify-between gap-3'>
                      <h3 className='text-sm font-semibold'>Metadata</h3>
                      <Badge className='bg-muted text-muted-foreground'>{formatSocial(selectedQuery.social)}</Badge>
                    </div>
                    <div className='mt-3 grid gap-3'>
                      <div className='grid gap-2 sm:grid-cols-[7rem_minmax(0,1fr)] sm:items-center'>
                        <Label htmlFor='sheet-query-name'>Name</Label>
                        <Input
                          id='sheet-query-name'
                          value={form.name}
                          readOnly={selectedQuery.status !== 'draft'}
                          className={selectedQuery.status === 'draft' ? 'text-black' : undefined}
                          onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
                        />
                      </div>
                      <div className='grid gap-2 sm:grid-cols-[7rem_minmax(0,1fr)] sm:items-center'>
                        <Label htmlFor='sheet-query-topic'>Source topic</Label>
                        <Textarea
                          id='sheet-query-topic'
                          rows={3}
                          value={form.topic}
                          readOnly={selectedQuery.status !== 'draft'}
                          className={`min-h-20 resize-none${selectedQuery.status === 'draft' ? ' text-black' : ''}`}
                          placeholder='Source topic used to generate this plan'
                          onChange={(event) => setForm((prev) => ({ ...prev, topic: event.target.value }))}
                        />
                      </div>
                    </div>
                  </section>

                  <section className='pt-4'>
                    <div className='flex items-center justify-between gap-3'>
                      <h3 className='text-sm font-semibold'>Search plan</h3>
                      {selectedQuery.social === 'linkedin' ? (
                        <span className='text-xs text-muted-foreground'>
                          {form.plan.queries?.length ?? DEFAULT_LINKEDIN_QUERY_INPUT_COUNT} LinkedIn content-search queries
                        </span>
                      ) : null}
                    </div>
                    <div className='mt-3 space-y-4'>
                      {selectedQuery.social === 'reddit' ? (
                        <>
                          <div className='space-y-2'>
                            <Label htmlFor='sheet-plan-topics'>Topic keywords</Label>
                            <Textarea
                              id='sheet-plan-topics'
                              rows={6}
                              value={toLines(form.plan.topics)}
                              readOnly={selectedQuery.status !== 'draft'}
                              className={selectedQuery.status === 'draft' ? 'text-black' : undefined}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  plan: {
                                    ...prev.plan,
                                    topics: parseLines(event.target.value),
                                  },
                                }))
                              }
                            />
                          </div>
                          <div className='space-y-2'>
                            <Label htmlFor='sheet-plan-subreddits'>Subreddits</Label>
                            <Textarea
                              id='sheet-plan-subreddits'
                              rows={6}
                              value={toLines(form.plan.subreddits)}
                              readOnly={selectedQuery.status !== 'draft'}
                              className={selectedQuery.status === 'draft' ? 'text-black' : undefined}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  plan: {
                                    ...prev.plan,
                                    subreddits: parseLines(event.target.value),
                                  },
                                }))
                              }
                            />
                          </div>
                          <div className='space-y-2'>
                            <Label htmlFor='sheet-plan-queries'>Queries</Label>
                            <Textarea
                              id='sheet-plan-queries'
                              rows={8}
                              value={toLines(form.plan.queries)}
                              readOnly={selectedQuery.status !== 'draft'}
                              className={selectedQuery.status === 'draft' ? 'text-black' : undefined}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  plan: {
                                    ...prev.plan,
                                    queries: parseLines(event.target.value),
                                  },
                                }))
                              }
                            />
                          </div>
                        </>
                      ) : (
                        <div className='space-y-3'>
                          <div className='space-y-1'>
                            <Label>LinkedIn queries</Label>
                            <p className='text-xs text-muted-foreground'>
                              Each row runs as its own LinkedIn content search.
                            </p>
                          </div>
                          <div className='space-y-2'>
                            {(form.plan.queries && form.plan.queries.length > 0 ? form.plan.queries : ['']).map((query, index) => (
                              <div key={index} className='grid gap-2 sm:grid-cols-[5rem_minmax(0,1fr)_auto] sm:items-center'>
                                <Label
                                  htmlFor={`sheet-plan-linkedin-query-${index}`}
                                  className='text-xs text-muted-foreground'
                                >
                                  Query {index + 1}
                                </Label>
                                <Input
                                  id={`sheet-plan-linkedin-query-${index}`}
                                  value={query}
                                  readOnly={selectedQuery.status !== 'draft'}
                                  className={selectedQuery.status === 'draft' ? 'text-black' : undefined}
                                  onChange={(event) => updateLinkedinQuery(index, event.target.value)}
                                />
                                {selectedQuery.status === 'draft' ? (
                                  <Button
                                    type='button'
                                    variant='ghost'
                                    size='icon'
                                    aria-label='Remove query'
                                    disabled={(form.plan.queries?.length ?? 1) <= 1}
                                    onClick={() => removeLinkedinQuery(index)}
                                  >
                                    <Trash2 className='size-4' />
                                  </Button>
                                ) : null}
                              </div>
                            ))}
                          </div>
                          {selectedQuery.status === 'draft' ? (
                            <Button type='button' variant='outline' size='sm' onClick={addLinkedinQuery}>
                              <Plus className='size-4' />
                              Add query
                            </Button>
                          ) : null}
                        </div>
                      )}
                    </div>
                  </section>
                </div>
              </>
            ) : null}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
