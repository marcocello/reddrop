import { createFileRoute } from '@tanstack/react-router'
import { SearchQueriesPage } from '@/features/search-queries'

export const Route = createFileRoute('/_authenticated/search-queries/')({
  component: SearchQueriesPage,
})
