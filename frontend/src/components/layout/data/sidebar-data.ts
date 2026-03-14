import { Command, KeyRound, ListTodo, PlayCircle, Settings, Users } from 'lucide-react'
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
      plan: 'Social Automation',
    },
  ],
  navGroups: [
    {
      title: 'Workflows',
      items: [
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
          title: 'Personas',
          url: '/personas',
          icon: Users,
        },
      ],
    },
    {
      title: 'Configuration',
      items: [
        {
          title: 'Auths',
          url: '/auths',
          icon: KeyRound,
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
