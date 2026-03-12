import { createFileRoute } from '@tanstack/react-router'
import { ThreadsPage } from '@/features/threads'

export const Route = createFileRoute('/_authenticated/threads/')({
  component: ThreadsPage,
})
