import { useState } from 'react'

export default function TransactionTable({ transactions }) {
  const [showFraudOnly, setShowFraudOnly] = useState(false)

  const filtered = showFraudOnly
    ? transactions.filter(t => t.actual_fraud === 1)
    : transactions

  const scoreColor = (score) => {
    if (score > 0.7) return '#ef4444'
    if (score > 0.3) return '#f59e0b'
    return '#22c55e'
  }

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ marginBottom: 0 }}>Transaction Stream (Sample)</h2>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13, color: '#8b8fa3' }}>
          <input
            type="checkbox"
            checked={showFraudOnly}
            onChange={e => setShowFraudOnly(e.target.checked)}
            style={{ accentColor: '#6366f1' }}
          />
          Show fraud only
        </label>
      </div>

      <div className="table-container" style={{ maxHeight: 400, overflowY: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Amount</th>
              <th>Category</th>
              <th>Fraud Score</th>
              <th>Prediction</th>
              <th>Actual</th>
              <th>Type</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 100).map((txn, i) => (
              <tr key={i} style={txn.actual_fraud ? { background: 'rgba(239, 68, 68, 0.05)' } : {}}>
                <td style={{ whiteSpace: 'nowrap', color: '#8b8fa3' }}>
                  {new Date(txn.timestamp).toLocaleDateString('en-US', {
                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                  })}
                </td>
                <td style={{ fontFamily: 'monospace' }}>${txn.amount.toFixed(2)}</td>
                <td>{txn.merchant_category}</td>
                <td>
                  <div className="score-bar">
                    <div
                      className="score-fill"
                      style={{
                        width: `${txn.fraud_score * 100}%`,
                        background: scoreColor(txn.fraud_score),
                      }}
                    />
                  </div>
                  <span style={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {txn.fraud_score.toFixed(3)}
                  </span>
                </td>
                <td>
                  <span className={`fraud-tag ${txn.predicted_fraud ? 'fraud' : 'legit'}`}>
                    {txn.predicted_fraud ? 'FLAGGED' : 'OK'}
                  </span>
                </td>
                <td>
                  <span className={`fraud-tag ${txn.actual_fraud ? 'fraud' : 'legit'}`}>
                    {txn.actual_fraud ? 'FRAUD' : 'LEGIT'}
                  </span>
                </td>
                <td style={{ color: '#8b8fa3', fontSize: 12 }}>
                  {txn.fraud_type !== 'none' ? txn.fraud_type : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
