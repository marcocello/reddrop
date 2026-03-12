import { Command, ListTodo, PlayCircle, Settings, Users } from 'lucide-react'
import { type SidebarData } from '../types'

export const sidebarData: SidebarData = {
  user: {
    name: 'reddrop',
    email: 'local@reddrop',
    avatar: '/images/favicon.png',
  },
  teams: [
    {
      name: 'Reddrop',
      logo: Command,
      plan: 'Search + Threads + Settings',
    },
  ],
  navGroups: [
    {
      title: 'Reddrop',
      items: [
        {
          title: 'Personas',
          url: '/personas',
          icon: Users,
        },
        {
          title: 'Jobs',
          url: '/jobs',
          icon: PlayCircle,
        },
        {
          title: 'Threads',
          url: '/threads',
          icon: ListTodo,
        },
        {
          title: 'Settings',
          url: '/settings',
          icon: Settings,
        },
      ],
    },
  ],
}
