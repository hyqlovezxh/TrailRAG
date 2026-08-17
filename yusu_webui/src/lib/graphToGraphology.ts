import Graph from 'graphology'
import type { GraphSubgraph } from '@/api/yusu'

/**
 * Deterministic palette used to color entity nodes by their type label.
 * The order is stable so the same entity type always gets the same color.
 */
export const ENTITY_COLORS: string[] = [
  '#10b981',
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
  '#f97316',
  '#84cc16',
  '#ec4899',
  '#6366f1',
  '#14b8a6',
  '#eab308'
]

/** Map an entity type label to a stable palette color (simple string hash). */
export const colorForType = (type: string): string => {
  let hash = 0
  for (let i = 0; i < type.length; i++) {
    hash = (hash * 31 + type.charCodeAt(i)) >>> 0
  }
  return ENTITY_COLORS[hash % ENTITY_COLORS.length]
}

/**
 * Convert a backend subgraph payload into a graphology graph.
 *
 * Pure function: when `target` is omitted a fresh directed multi-graph is
 * created; otherwise nodes/edges are merged into `target` (existing keys are
 * left untouched). Node attributes are remapped for sigma rendering:
 * `label` -> display name, `category` -> entity type label, `color` ->
 * stable per-category color. Edges whose endpoints are missing are skipped.
 */
export function graphToGraphology(subgraph: GraphSubgraph, target?: Graph): Graph {
  const graph = target ?? new Graph({ multi: true, type: 'directed' })

  // IMPORTANT: Sigma observes the graphology graph *live* and validates a
  // node's `x`/`y` the instant `graph.addNode` runs. If coordinates are only
  // assigned afterward, Sigma throws "could not find a valid position" on the
  // very first inserted node. So we compute every new node's coordinates
  // FIRST, then add nodes already carrying `x`/`y`.
  const isFresh = graph.order === 0

  // Nodes that still need to be inserted into `graph` (skips duplicates).
  const toAdd = subgraph.nodes.filter(
    (n) => n.entity_id && !graph.hasNode(n.entity_id)
  )
  const coords = new Map<string, { x: number; y: number }>()

  if (isFresh) {
    // Brand-new graph: deterministic disc layout keyed by node id so placement
    // is identical on every reload and connected nodes spread out organically.
    const n = toAdd.length
    const baseR = Math.max(120, Math.sqrt(n) * 60)
    toAdd.forEach((node, i) => {
      const id = node.entity_id as string
      const angle = (i / Math.max(1, n)) * Math.PI * 2
      const r = baseR * (0.45 + 0.55 * hash01(id))
      coords.set(id, { x: Math.cos(angle) * r, y: Math.sin(angle) * r })
    })
  } else {
    // Incremental merge: existing nodes keep their positions; only the newly
    // inserted ones get a (random) spot near the origin.
    toAdd.forEach((node) => {
      const id = node.entity_id as string
      coords.set(id, {
        x: (Math.random() - 0.5) * 200,
        y: (Math.random() - 0.5) * 200
      })
    })
  }

  for (const node of subgraph.nodes) {
    if (!node.entity_id || graph.hasNode(node.entity_id)) continue
    const category = node.label || node.type || 'Entity'
    const c = coords.get(node.entity_id)!
    graph.addNode(node.entity_id, {
      // NOTE: Sigma v3 uses `type` as the *renderer program* selector
      // ("circle", "image", ...). The backend payload uses `type` for the
      // semantic entity category, so we must NOT forward it as Sigma's `type`.
      // We pin the program to "circle" and keep the category in `category`.
      ...node,
      label: node.name || node.entity_id,
      category,
      color: colorForType(category),
      size: 8,
      x: c.x,
      y: c.y,
      type: 'circle'
    })
  }

  for (const edge of subgraph.edges) {
    const source = edge.source_id
    const targetId = edge.target_id
    if (!source || !targetId) continue
    if (!graph.hasNode(source) || !graph.hasNode(targetId)) continue
    if (graph.hasEdge(source, targetId)) continue
    graph.addEdge(source, targetId, {
      // Same as nodes: `edge.type` is the semantic relation type, not the
      // Sigma edge program. Pin the program to "arrow" (directed graph) and
      // surface the relation in `label`.
      ...edge,
      label: edge.type || edge.text || '',
      type: 'arrow',
      color: '#94a3b8',
      size: 1
    })
  }

  // Defensive final pass: guarantee no node is left without numeric coordinates
  // even if a caller mutates the graph outside this function. Newly placed
  // nodes lie on a golden-angle spiral keyed by insertion order.
  let spiral = 0
  graph.forEachNode((n) => {
    const x = graph.getNodeAttribute(n, 'x')
    const y = graph.getNodeAttribute(n, 'y')
    if (typeof x !== 'number' || typeof y !== 'number') {
      const radius = 10 * Math.sqrt(spiral)
      const angle = spiral * 2.399963229728653
      graph.setNodeAttribute(n, 'x', radius * Math.cos(angle))
      graph.setNodeAttribute(n, 'y', radius * Math.sin(angle))
      spiral++
    }
  })

  return graph
}

/** FNV-1a hash -> float in [0, 1). Used for stable, deterministic placement. */
function hash01(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return ((h >>> 0) % 100000) / 100000
}
