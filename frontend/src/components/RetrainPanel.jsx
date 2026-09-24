import { useState } from 'react'

export default function RetrainPanel({ overview, onRetrained }) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const needsRetrain = overview?.retrain_urgency !== 'none'

  const handleRetrain = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/retrain', { method: 'POST' })
      const data = await res.json()
      setResult(data)
      onRetrained(data)
    } catch (err) {
      console.error(err)
    }
    setLoading(false)
  }

  return (
    <div className={`retrain-card ${needsRetrain ? 'needs-retrain' : ''}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h2 style={{ fontSize: 14, fontWeight: 600, color: '#8b8fa3', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            Retraining Loop
          </h2>
          {needsRetrain && !result ? (
            <div>
              <p style={{ color: '#f59e0b', fontWeight: 500, marginBottom: 6 }}>
                Retrain recommended ({overview.retrain_urgency})
              </p>
              {overview.retrain_reasons?.map((r, i) => (
                <p key={i} style={{ color: '#8b8fa3', fontSize: 13 }}>- {r}</p>
              ))}
            </div>
          ) : result ? (
            <p style={{ color: '#22c55e', fontWeight: 500 }}>Model retrained successfully</p>
          ) : (
            <p style={{ color: '#22c55e', fontWeight: 500 }}>Model is up to date</p>
          )}
        </div>
        <button
          className="retrain-btn"
          onClick={handleRetrain}
          disabled={loading || result}
        >
          {loading ? 'Retraining...' : result ? 'Retrained' : 'Retrain Now'}
        </button>
      </div>

      {result && (
        <div className="retrain-result">
          <div className="retrain-col">
            <h3>Before (Old Model)</h3>
            <div className="retrain-metric">
              <span className="metric-label">AUPRC</span>
              <span>{result.before.auprc}</span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Precision</span>
              <span>{result.before.precision}</span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Recall</span>
              <span>{result.before.recall}</span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Train Period</span>
              <span style={{ fontSize: 11 }}>{result.before.train_period}</span>
            </div>
          </div>
          <div className="retrain-col">
            <h3>After (Retrained)</h3>
            <div className="retrain-metric">
              <span className="metric-label">AUPRC</span>
              <span>
                {result.after.auprc}
                <span className={result.improvement.auprc_delta >= 0 ? 'delta-positive' : 'delta-negative'}
                      style={{ marginLeft: 8, fontSize: 12 }}>
                  {result.improvement.auprc_delta >= 0 ? '+' : ''}{result.improvement.auprc_delta}
                </span>
              </span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Precision</span>
              <span>
                {result.after.precision}
                <span className={result.improvement.precision_delta >= 0 ? 'delta-positive' : 'delta-negative'}
                      style={{ marginLeft: 8, fontSize: 12 }}>
                  {result.improvement.precision_delta >= 0 ? '+' : ''}{result.improvement.precision_delta}
                </span>
              </span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Recall</span>
              <span>
                {result.after.recall}
                <span className={result.improvement.recall_delta >= 0 ? 'delta-positive' : 'delta-negative'}
                      style={{ marginLeft: 8, fontSize: 12 }}>
                  {result.improvement.recall_delta >= 0 ? '+' : ''}{result.improvement.recall_delta}
                </span>
              </span>
            </div>
            <div className="retrain-metric">
              <span className="metric-label">Train Period</span>
              <span style={{ fontSize: 11 }}>{result.after.train_period}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
