import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend
} from 'recharts'

export default function MetricsChart({ data, baseline }) {
  const formatted = data.map(d => ({
    ...d,
    date: new Date(d.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
  }))

  return (
    <div className="card">
      <h2>Model Performance Over Time</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={formatted}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e3a" />
          <XAxis dataKey="date" tick={{ fill: '#8b8fa3', fontSize: 11 }} />
          <YAxis domain={[0, 1.05]} tick={{ fill: '#8b8fa3', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#1a1d27', border: '1px solid #2a2e3a', borderRadius: 8 }}
            labelStyle={{ color: '#e1e4ea' }}
          />
          <Legend />
          {baseline && (
            <ReferenceLine
              y={baseline}
              stroke="#22c55e"
              strokeDasharray="6 3"
              label={{ value: 'Baseline', fill: '#22c55e', fontSize: 11 }}
            />
          )}
          <Line type="monotone" dataKey="auprc" stroke="#ef4444" strokeWidth={2}
                dot={{ r: 3 }} name="AUPRC" />
          <Line type="monotone" dataKey="precision" stroke="#3b82f6" strokeWidth={1.5}
                dot={{ r: 2 }} name="Precision" />
          <Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={1.5}
                dot={{ r: 2 }} name="Recall" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
