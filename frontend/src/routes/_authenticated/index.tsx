import { createFileRoute } from '@tanstack/react-router'
import { JobsPage } from '@/features/jobs'

export const Route = createFileRoute('/_authenticated/')({
  component: JobsPage,
})
