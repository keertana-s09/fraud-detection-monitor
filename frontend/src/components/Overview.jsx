export default function Overview({ data }) {
  if (!data) return null

  return (
    <div className="overview-grid">
      <div className="stat-card blue">
        <div className="label">Total Transactions</div>
        <div className="value">{(data.total_transactions / 1000).toFixed(0)}K</div>
        <div className="detail">{data.train_period}</div>
      </div>
      <div className="stat-card red">
        <div className="label">Fraud Rate</div>
        <div className="value">{data.fraud_rate}%</div>
        <div className="detail">{data.total_fraud.toLocaleString()} fraudulent txns</div>
      </div>
      <div className="stat-card green">
        <div className="label">Baseline AUPRC</div>
        <div className="value">{data.baseline_auprc}</div>
        <div className="detail">Recall: {data.baseline_recall}</div>
      </div>
      <div className="stat-card accent">
        <div className="label">Model</div>
        <div className="value" style={{fontSize: 16}}>{data.model_type}</div>
        <div className="detail">Precision: {data.baseline_precision}</div>
      </div>
      <div className={`stat-card ${data.total_alerts > 5 ? 'yellow' : 'green'}`}>
        <div className="label">Drift Alerts</div>
        <div className="value">{data.total_alerts}</div>
        <div className="detail">{data.critical_alerts} critical</div>
      </div>
      <div className={`stat-card ${data.retrain_urgency === 'immediate' ? 'red' : data.retrain_urgency === 'scheduled' ? 'yellow' : 'green'}`}>
        <div className="label">Retrain Status</div>
        <div className="value" style={{fontSize: 18, textTransform: 'capitalize'}}>{data.retrain_urgency}</div>
        <div className="detail">Monitoring {data.prod_period}</div>
      </div>
    </div>
  )
}
