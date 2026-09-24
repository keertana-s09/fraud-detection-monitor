import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'

export default function FraudMixChart({ data }) {
  return (
    <div className="card">
      <h2>Fraud Type Composition Over Time</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e3a" />
          <XAxis dataKey="month" tick={{ fill: '#8b8fa3', fontSize: 11 }} />
          <YAxis tick={{ fill: '#8b8fa3', fontSize: 11 }} unit="%" />
          <Tooltip
            contentStyle={{ background: '#1a1d27', border: '1px solid #2a2e3a', borderRadius: 8 }}
            labelStyle={{ color: '#e1e4ea' }}
            formatter={(val) => [`${val}%`]}
          />
          <Legend />
          <Bar dataKey="cnp" stackId="a" fill="#ef4444" name="CNP" />
          <Bar dataKey="ato" stackId="a" fill="#3b82f6" name="ATO" />
          <Bar dataKey="bust_out" stackId="a" fill="#22c55e" name="Bust-out" />
          <Bar dataKey="micro_test" stackId="a" fill="#a855f7" name="Micro-test" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
