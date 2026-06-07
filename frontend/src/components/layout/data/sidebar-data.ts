import { faSatelliteDish } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { createElement } from 'react'
import { KeyRound, ListTodo, PlayCircle, Route, Settings, Users } from 'lucide-react'
import { type SidebarData } from '../types'

const SatelliteDishLogo = ({ className }: { className?: string }) =>
  createElement(FontAwesomeIcon, { icon: faSatelliteDish, className })

export const sidebarData: SidebarData = {
  user: {
    name: 'reddrop',
    email: 'local@reddrop',
    avatar: '/images/favicon.svg',
  },
  teams: [
    {
      name: 'RiteSignal',
      logo: SatelliteDishLogo,
      plan: 'Conversation Automation',
    },
  ],
  navGroups: [
    {
      title: 'Workflows',
      items: [
        {
          title: 'Personas',
          url: '/personas',
          icon: Users,
        },
        {
          title: 'Search Queries',
          url: '/search-queries',
          icon: Route,
        },
        {
          title: 'Jobs',
          url: '/jobs',
          icon: PlayCircle,
        },
      ],
    },
    {
      title: 'Results',
      items: [
        {
          title: 'Conversations',
          url: '/threads',
          icon: ListTodo,
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
