import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer
} from 'recharts'

export default function DriftChart({ data }) {
  const formatted = data.map(d => ({
    ...d,
    date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  return (
    <div className="card">
      <h2>Drift Score (PSI) Over Time</h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e3a" />
          <XAxis dataKey="date" tick={{ fill: '#8b8fa3', fontSize: 11 }} />
          <YAxis tick={{ fill: '#8b8fa3', fontSize: 11 }} domain={[0, 'auto']} />
          <Tooltip
            contentStyle={{ background: '#1a1d27', border: '1px solid #2a2e3a', borderRadius: 8 }}
            labelStyle={{ color: '#e1e4ea' }}
            formatter={(val) => [val.toFixed(4), 'PSI']}
          />
          <ReferenceLine
            y={0.1}
            stroke="#f59e0b"
            strokeDasharray="6 3"
            label={{ value: 'Warning', fill: '#f59e0b', fontSize: 11, position: 'right' }}
          />
          <ReferenceLine
            y={0.25}
            stroke="#ef4444"
            strokeDasharray="6 3"
            label={{ value: 'Critical', fill: '#ef4444', fontSize: 11, position: 'right' }}
          />
          <Area
            type="monotone"
            dataKey="psi"
            stroke="#a855f7"
            fill="rgba(168, 85, 247, 0.15)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
