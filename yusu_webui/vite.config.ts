import { defineConfig, loadEnv, type Plugin } from 'vite'
import path from 'path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Use relative imports here. The '@' alias is configured in resolve.alias
// below and only takes effect during bundling — Node cannot resolve it when
// loading vite.config.ts. Bun resolves tsconfig paths natively, masking the
// issue, but Node does not.
import { normalizeApiPrefix, normalizeWebuiPrefix } from './src/lib/pathPrefix.ts'

/**
 * Inject `<script>window.__YUSU_CONFIG__ = ...</script>` into index.html.
 *
 * This mirrors what the FastAPI server does at request time in production
 * (see `SmartStaticFiles._inject_runtime_config` in
 * `yusu/api/yusu_server.py`). Doing it in dev too means the SPA
 * always reads its prefix the same way, so behaviour matches between
 * `bun run dev` and a production deploy.
 *
 * Only `VITE_DEV_API_PREFIX` is read; the WebUI mount path is fixed at
 * `/webui` (matching the backend's hardcoded `WEBUI_PATH`), so the
 * injected `webuiPrefix` follows the production formula
 * `apiPrefix + "/webui/"` automatically.
 */
function yusuRuntimeConfigPlugin(env: Record<string, string>): Plugin {
  const apiPrefix = normalizeApiPrefix(env.VITE_DEV_API_PREFIX)
  const webuiPrefix = normalizeWebuiPrefix(apiPrefix ? `${apiPrefix}/webui/` : '')
  const payload = JSON.stringify({ apiPrefix, webuiPrefix }).replace(
    /<\//g,
    '<\\/'
  )
  const snippet = `<script>window.__YUSU_CONFIG__ = ${payload};</script>`

  return {
    name: 'yusu-dev-runtime-config',
    apply: 'serve',
    transformIndexHtml(html: string) {
      return html.replace('<!-- __YUSU_RUNTIME_CONFIG__ -->', snippet)
    }
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // Dev-only: prefix every proxied endpoint with the simulated site
  // prefix so e.g. `/site01/documents/...` is forwarded to the backend
  // running with YUSU_API_PREFIX=/site01.
  const devApiPrefix = normalizeApiPrefix(env.VITE_DEV_API_PREFIX)

  return {
    plugins: [react(), tailwindcss(), yusuRuntimeConfigPlugin(env)],
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, './src')
      },
      // Force all modules to use the same katex instance
      // This ensures mhchem extension registered in main.tsx is available to rehype-katex
      dedupe: ['katex']
    },
    // Relative base: asset URLs in index.html become `./assets/...` so the
    // built bundle works under any reverse-proxy mount point. The browser
    // resolves them against the current document URL — which means the
    // server MUST serve index.html at a URL ending in '/' (the existing
    // /webui → /webui/ redirect already handles this).
    base: './',
    build: {
      outDir: path.resolve(import.meta.dirname, './dist'),
      emptyOutDir: true,
      chunkSizeWarningLimit: 3800,
      rollupOptions: {
        // Group the graphology/sigma/@react-sigma stack into a single chunk so
        // the browser fetches them once and the Sigma<->graphology live
        // bindings stay in one module scope. (Historical note: an earlier
        // "Cannot access 'X' before initialization" crash here was NOT a
        // bundler bug — it was a forward-reference TDZ in GraphView.tsx where
        // `loadFullGraph` was read in a useEffect dependency array before its
        // `const` declaration. That is fixed at the source level.)
        output: {
          manualChunks(id) {
            if (
              id.includes('node_modules/graphology') ||
              id.includes('node_modules/sigma') ||
              id.includes('node_modules/@react-sigma') ||
              id.includes('node_modules/@sigma')
            ) {
              return 'graph-vendor'
            }
          },
          // Ensure consistent chunk naming format
          chunkFileNames: 'assets/[name]-[hash].js',
          // Entry file naming format
          entryFileNames: 'assets/[name]-[hash].js',
          // Asset file naming format
          assetFileNames: 'assets/[name]-[hash].[ext]'
        }
      }
    },
    server: {
      proxy: env.VITE_API_PROXY === 'true' && env.VITE_API_ENDPOINTS ?
        Object.fromEntries(
          env.VITE_API_ENDPOINTS.split(',').map(endpoint => [
            devApiPrefix + endpoint,
            {
              target: env.VITE_BACKEND_URL || 'http://localhost:9621',
              changeOrigin: true
              // No rewrite: the backend already understands its own prefix
              // via FastAPI's root_path, so forward the path verbatim.
            }
          ])
        ) : {}
    }
  }
})
