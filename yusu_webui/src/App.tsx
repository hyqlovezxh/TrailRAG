import { useEffect } from 'react'
import TabVisibilityProvider from '@/contexts/TabVisibilityProvider'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import SiteHeader from '@/features/SiteHeader'
import KnowledgeBaseView from '@/features/KnowledgeBaseView'
import ChatView from '@/features/ChatView'
import ModelManageView from '@/features/ModelManageView'
import GraphView from '@/features/GraphView'
import EvalView from '@/features/EvalView'
import { Tabs, TabsContent } from '@/components/ui/Tabs'
import ErrorBoundary from '@/components/ErrorBoundary'

function App() {
  const currentTab = useSettingsStore.use.currentTab()

  useEffect(() => {
    const performHealthCheck = () => {
      useBackendState.getState().check()
    }
    useBackendState.getState().setHealthCheckFunction(performHealthCheck)
    useBackendState.getState().resetHealthCheckTimer()

    return () => {
      useBackendState.getState().clearHealthCheckTimer()
    }
  }, [])

  return (
    <TabVisibilityProvider>
      <main className="flex h-screen w-screen overflow-hidden">
        <Tabs
          defaultValue={currentTab}
          className="!m-0 flex grow flex-col !p-0 overflow-hidden"
          onValueChange={(tab) => useSettingsStore.getState().setCurrentTab(tab as never)}
        >
          <SiteHeader />
          <div className="relative grow">
            <TabsContent value="knowledge" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
              <ErrorBoundary>
                <KnowledgeBaseView />
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="chat" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
              <ErrorBoundary>
                <ChatView />
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="models" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
              <ErrorBoundary>
                <ModelManageView />
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="graph" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
              <ErrorBoundary>
                <GraphView />
              </ErrorBoundary>
            </TabsContent>
            <TabsContent value="eval" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
              <ErrorBoundary>
                <EvalView />
              </ErrorBoundary>
            </TabsContent>
          </div>
        </Tabs>
      </main>
    </TabVisibilityProvider>
  )
}

export default App