import { createFileRoute } from '@tanstack/react-router'
import { PersonasPage } from '@/features/personas'

export const Route = createFileRoute('/_authenticated/personas/')({
  component: PersonasPage,
})
