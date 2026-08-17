import Button from '@/components/ui/Button'
import { SiteInfo } from '@/lib/constants'
import { TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { useSettingsStore } from '@/stores/settings'
import { useBackendState } from '@/stores/state'
import { cn } from '@/lib/utils'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { ZapIcon, LogOutIcon } from 'lucide-react'
import LanguageToggle from '@/components/LanguageToggle'
import ThemeToggle from '@/components/ThemeToggle'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/Tooltip'

interface NavigationTabProps {
  value: string
  currentTab: string
  children: React.ReactNode
}

function NavigationTab({ value, currentTab, children }: NavigationTabProps) {
  return (
    <TabsTrigger
      value={value}
      className={cn(
        'cursor-pointer px-2 py-1 transition-all',
        currentTab === value ? '!bg-emerald-400 !text-zinc-50' : 'hover:bg-background/60'
      )}
    >
      {children}
    </TabsTrigger>
  )
}

function TabsNavigation() {
  const currentTab = useSettingsStore.use.currentTab()
  const { t } = useTranslation()

  return (
    <div className="flex h-8 self-center">
      <TabsList className="h-full gap-2">
        <NavigationTab value="knowledge" currentTab={currentTab}>
          {t('header.knowledge')}
        </NavigationTab>
        <NavigationTab value="chat" currentTab={currentTab}>
          {t('header.chat')}
        </NavigationTab>
        <NavigationTab value="models" currentTab={currentTab}>
          {t('header.models')}
        </NavigationTab>
        <NavigationTab value="graph" currentTab={currentTab}>
          {t('header.graph')}
        </NavigationTab>
        <NavigationTab value="eval" currentTab={currentTab}>
          {t('header.eval')}
        </NavigationTab>
      </TabsList>
    </div>
  )
}

function HealthIndicator() {
  const health = useBackendState.use.health()
  const message = useBackendState.use.message()
  const { t } = useTranslation()

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="flex cursor-default items-center">
            <span
              className={cn(
                'size-2 rounded-full',
                health ? 'bg-emerald-400' : 'bg-destructive'
              )}
            />
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {health ? t('header.connected') : message || t('header.disconnected')}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export default function SiteHeader() {
  const { t } = useTranslation()
  const navigate = useNavigate()

  const handleDisconnect = () => {
    useSettingsStore.getState().setBaseUrl('')
    useSettingsStore.getState().setApiKey(null)
    navigate('/login')
  }

  return (
    <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50 flex h-10 w-full border-b px-4 backdrop-blur">
      <div className="flex min-w-[200px] items-center">
        <a href={SiteInfo.home} className="flex items-center gap-2">
          <ZapIcon className="size-4 text-emerald-400" aria-hidden="true" />
          <span className="font-bold md:inline-block">{SiteInfo.name}</span>
        </a>
      </div>

      <div className="flex h-10 flex-1 items-center justify-center">
        <TabsNavigation />
      </div>

      <nav className="flex w-[200px] items-center justify-end">
        <div className="flex items-center gap-2">
          <HealthIndicator />
          <LanguageToggle />
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon"
            side="bottom"
            tooltip={t('header.disconnect')}
            onClick={handleDisconnect}
          >
            <LogOutIcon className="size-4" aria-hidden="true" />
          </Button>
        </div>
      </nav>
    </header>
  )
}