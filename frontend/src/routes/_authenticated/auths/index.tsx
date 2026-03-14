import { createFileRoute } from '@tanstack/react-router'
import { AuthsPage } from '@/features/auths'

export const Route = createFileRoute('/_authenticated/auths/')({
  component: AuthsPage,
})
