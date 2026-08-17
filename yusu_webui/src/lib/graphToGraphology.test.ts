import { describe, expect, it } from 'bun:test'
import Graph from 'graphology'
import { colorForType, ENTITY_COLORS, graphToGraphology } from './graphToGraphology'
import type { GraphSubgraph } from '@/api/yusu'

const sampleSubgraph = (): GraphSubgraph => ({
  nodes: [
    {
      entity_id: 'ent-zhangsan',
      type: 'entity',
      name: '张三',
      label: 'Person',
      normalized_name: 'zhangsan',
      description: '一个自然人'
    },
    {
      entity_id: 'ent-lisi',
      type: 'entity',
      name: '李四',
      label: 'Person'
    },
    {
      entity_id: 'ent-company',
      type: 'entity',
      name: '某公司',
      label: 'Organization'
    }
  ],
  edges: [
    {
      source_id: 'ent-zhangsan',
      target_id: 'ent-lisi',
      triple_id: 't1',
      type: '资金转账',
      text: '张三向李四转账',
      file_ids: ['f1']
    },
    {
      source_id: 'ent-zhangsan',
      target_id: 'ent-company',
      type: '任职',
      text: '张三任职于某公司'
    },
    {
      source_id: 'ent-missing',
      target_id: 'ent-lisi',
      type: '未知关系'
    }
  ]
})

describe('colorForType', () => {
  it('is deterministic for the same type', () => {
    expect(colorForType('Person')).toBe(colorForType('Person'))
    expect(colorForType('Organization')).toBe(colorForType('Organization'))
  })

  it('always returns a palette color', () => {
    for (const type of ['Person', 'Organization', '', '地点', '金额', '事件']) {
      expect(ENTITY_COLORS).toContain(colorForType(type))
    }
  })
})

describe('graphToGraphology', () => {
  it('creates a directed multi-graph from a subgraph payload', () => {
    const graph = graphToGraphology(sampleSubgraph())
    expect(graph.order).toBe(3)
    expect(graph.size).toBe(2)
    expect(graph.type).toBe('directed')
    expect(graph.multi).toBe(true)
  })

  it('maps node attributes for sigma rendering', () => {
    const graph = graphToGraphology(sampleSubgraph())
    const attrs = graph.getNodeAttributes('ent-zhangsan')
    expect(attrs.label).toBe('张三')
    expect(attrs.category).toBe('Person')
    expect(attrs.color).toBe(colorForType('Person'))
    expect(attrs.entity_id).toBe('ent-zhangsan')
  })

  it('skips edges whose endpoints are not in the graph', () => {
    const graph = graphToGraphology(sampleSubgraph())
    const edgeAttrs = graph.getEdgeAttributes(graph.edges()[0])
    expect(edgeAttrs.label).toBe('资金转账')
    expect(graph.hasEdge('ent-missing', 'ent-lisi')).toBe(false)
  })

  it('merges into an existing graph without duplicating nodes/edges', () => {
    const existing = new Graph({ multi: true, type: 'directed' })
    existing.addNode('ent-zhangsan', { label: '张三', category: 'Person', color: '#000000' })
    graphToGraphology(sampleSubgraph(), existing)
    expect(existing.order).toBe(3)
    expect(existing.size).toBe(2)
    // existing node attributes are left untouched
    expect(existing.getNodeAttribute('ent-zhangsan', 'color')).toBe('#000000')
  })

  it('falls back to entity_id when name is missing', () => {
    const graph = graphToGraphology({
      nodes: [{ entity_id: 'e1', type: 'entity', name: '', label: '' }],
      edges: []
    })
    expect(graph.getNodeAttribute('e1', 'label')).toBe('e1')
  })
})
