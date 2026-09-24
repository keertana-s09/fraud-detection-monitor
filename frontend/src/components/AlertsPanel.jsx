import { useState } from 'react'

export default function AlertsPanel({ alerts }) {
  const [filter, setFilter] = useState('all')

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter(a => a.severity === filter)

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ marginBottom: 0 }}>Drift & Degradation Alerts</h2>
        <div style={{ display: 'flex', gap: 8 }}>
          {['all', 'critical', 'warning'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '4px 12px',
                borderRadius: 6,
                border: `1px solid ${filter === f ? '#6366f1' : '#2a2e3a'}`,
                background: filter === f ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
                color: filter === f ? '#818cf8' : '#8b8fa3',
                cursor: 'pointer',
                fontSize: 12,
                textTransform: 'capitalize',
              }}
            >
              {f} {f !== 'all' && `(${alerts.filter(a => a.severity === f).length})`}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {filtered.length === 0 ? (
          <p style={{ color: '#8b8fa3', textAlign: 'center', padding: 20 }}>No alerts</p>
        ) : (
          filtered.map((alert, i) => (
            <div key={i} className={`alert-item ${alert.severity}`}>
              <span className={`alert-badge ${alert.severity}`}>
                {alert.severity}
              </span>
              <span className="alert-message">
                <strong>{alert.type}</strong>: {alert.message}
              </span>
              <span className="alert-time">
                {new Date(alert.timestamp).toLocaleDateString('en-US', {
                  month: 'short', day: 'numeric'
                })}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
