// OptChat / UniiChat — desktop half of the unified plugin package.
// Live memory-engine status (pane + status-bar chip) and a full view browser
// (page + sidebar row + palette command). Talks to dashboard/plugin_api.py
// through ctx.rest (namespaced /api/plugins/optchat). No JSX: jsx/jsxs only.
import { jsx, jsxs } from 'react/jsx-runtime'
import {
  PALETTE_AREA, ROUTES_AREA, SIDEBAR_NAV_AREA, STATUSBAR_AREAS,
  host, useQuery, useValue,
} from '@hermes/plugin-sdk'

const POLL_MS = 4000
const mono = { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }
const dim = { color: 'var(--muted-foreground)' }

function fmtBytes(n) {
  if (n == null) return '—'
  if (n < 1024) return n + ' B'
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB'
  return (n / 1024 / 1024).toFixed(2) + ' MB'
}

function useStatus(ctx, sid) {
  return useQuery({
    queryKey: ['optchat', 'status', sid || ''],
    queryFn: () => ctx.rest('/status' + (sid ? '?session=' + encodeURIComponent(sid) : '')),
    refetchInterval: POLL_MS,
    retry: 0,
  })
}

function sawtooth(st) {
  if (!st || !st.high) return 0
  return Math.min(1, (st.view_bytes || 0) / st.high)
}

function Chip(ctx) {
  const sid = useValue(host.state.activeSessionId)
  const { data } = useStatus(ctx, sid)
  const eng = data && data.engine
  const live = eng && eng.engine_selected && eng.plugin_enabled
  const color = !live ? 'var(--muted-foreground)' : (data.pending > 0 ? '#e8b93e' : '#3fb950')
  return jsxs('button', {
    title: 'OptChat memory — ' + (data ? 'T ' + data.T + ', view ' + fmtBytes(data.view_bytes) : 'idle'),
    onClick: () => host.navigate('/optchat'),
    style: { display: 'inline-flex', alignItems: 'center', gap: 5, border: 0, background: 'transparent', cursor: 'pointer', fontSize: 11, color: 'var(--muted-foreground)', padding: '0 4px' },
  }, [
    jsx('span', { style: { width: 7, height: 7, borderRadius: 9, background: color, display: 'inline-block' } }, 'dot'),
    jsx('span', { style: mono }, 'mem ' + (data ? fmtBytes(data.view_bytes) : '·')),
  ])
}

function Stat({ label, value }) {
  return jsxs('div', { style: { display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12, padding: '2px 0' } }, [
    jsx('span', { style: dim }, label + 'l'),
    jsx('span', { style: mono }, String(value)),
  ])
}

function StatusBody({ ctx, compact }) {
  const sid = useValue(host.state.activeSessionId)
  const { data, isLoading, error } = useStatus(ctx, sid)
  const st = data
  const eng = st && st.engine
  const pct = Math.round(sawtooth(st) * 100)
  return jsxs('div', { style: { padding: 10, display: 'flex', flexDirection: 'column', gap: 8, minWidth: 0 } }, [
    jsxs('div', { style: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 } }, [
      jsx('span', { style: { width: 8, height: 8, borderRadius: 9, background: eng && eng.engine_selected && eng.plugin_enabled ? '#3fb950' : '#888', display: 'inline-block' } }, 'd'),
      jsx('strong', {}, eng && eng.engine_selected ? 'engine active' : 'engine not selected'),
    ]),
    eng && !eng.engine_selected
      ? jsx('div', { style: { fontSize: 11, ...dim } }, 'Run: hermes config set context.engine optchat (and enable the plugin)')
      : null,
    st ? jsxs('div', {}, [
      jsxs('div', { title: 'view sawtooth (64→128 KB)', style: { height: 6, borderRadius: 3, background: 'var(--border)', overflow: 'hidden', marginBottom: 6 } }, [
        jsx('div', { style: { height: '100%', width: pct + '%', background: 'var(--accent)' } }, 'bar'),
      ]),
      Stat('messages (T)', st.T),
      Stat('covered by view', st.covered),
      Stat('view', fmtBytes(st.view_bytes) + ' · ' + st.view_lines + ' lines'),
      compact ? null : Stat('lines per level', Object.entries(st.view_levels || {}).map(([l, n]) => l + ':' + n).join('  ')),
      Stat('built nodes', st.built_nodes),
      Stat('compactions', st.compactions),
      Stat('building now', st.pending),
      Stat('session', st.session),
    ]) : jsx('div', { style: { fontSize: 12, ...dim } }, isLoading ? 'loading…' : error ? 'no data (engine idle?)' : 'idle'),
  ])
}

function ViewLine({ ctx, line }) {
  const q = useQuery({
    queryKey: ['optchat', 'line', line.l, line.i],
    queryFn: () => ctx.rest('/line?l=' + line.l + '&i=' + line.i),
    enabled: false,
    retry: 0,
  })
  return jsxs('details', { style: { borderBottom: '1px solid var(--border)', padding: '3px 0' } }, [
    jsxs('summary', { style: { cursor: 'pointer', fontSize: 12, display: 'flex', gap: 8, minWidth: 0 } }, [
      jsx('span', { style: { ...mono, ...dim, flex: '0 0 70px' } }, line.name),
      jsx('span', { style: { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, '(tap to load)'),
    ]),
    jsx('div', { style: { padding: '4px 0 4px 78px', fontSize: 12, whiteSpace: 'pre-wrap' } },
      q.isFetching || q.data ? ((q.data && q.data.text) || '…') : jsx('a', { style: { cursor: 'pointer', ...dim }, onClick: () => q.refetch() }, 'load line') ),
  ])
}

function Page(ctx) {
  const sid = useValue(host.state.activeSessionId)
  const { data: st } = useStatus(ctx, sid)
  const view = useQuery({
    queryKey: ['optchat', 'view', sid || '', (st && st.view_lines) || 0],
    queryFn: () => ctx.rest('/view?tail=400' + (sid ? '&session=' + encodeURIComponent(sid) : '')),
    refetchInterval: POLL_MS,
    retry: 0,
  })
  return jsxs('div', { style: { display: 'flex', gap: 16, padding: 16, alignItems: 'flex-start', flexWrap: 'wrap' } }, [
    jsx('div', { style: { flex: '0 0 260px' } }, jsx(StatusBody, { ctx, compact: false })),
    jsx('div', { style: { flex: '1 1 420px', minWidth: 320, maxHeight: '70vh', overflowY: 'auto' } }, [
      jsx('div', { style: { fontSize: 12, ...dim, marginBottom: 6 } },
        'view — ' + ((view.data && view.data.lines && view.data.lines.length) || 0) + ' lines shown (most recent)'),
      ...(((view.data && view.data.lines) || []).slice().reverse().map((ln) =>
        jsx(ViewLine, { ctx, line: ln }, ln.name))),
    ]),
  ])
}

export default {
  id: 'optchat',
  name: 'OptChat / UniiChat memory',
  defaultEnabled: false,
  register(ctx) {
    ctx.register({
      id: 'pane', area: 'panes', title: 'OptChat',
      data: { placement: 'right', width: '280px' },
      render: () => jsx(StatusBody, { ctx, compact: true }),
    })
    ctx.register({ id: 'page', area: ROUTES_AREA, data: { path: '/optchat' }, render: () => jsx(Page, { ctx }) })
    ctx.register({ id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: '/optchat', label: 'Memory', codicon: 'database' } })
    ctx.register({ id: 'cmd', area: PALETTE_AREA, data: { label: 'OptChat: open memory view', run: () => host.navigate('/optchat') } })
    ctx.register({ id: 'chip', area: STATUSBAR_AREAS.right, render: () => jsx(Chip, { ctx }) })
  },
}