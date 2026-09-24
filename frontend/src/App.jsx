import { useState, useEffect } from 'react'
import UploadPage from './components/UploadPage'
import Overview from './components/Overview'
import MetricsChart from './components/MetricsChart'
import DriftChart from './components/DriftChart'
import AlertsPanel from './components/AlertsPanel'
import FraudMixChart from './components/FraudMixChart'
import FeatureImportance from './components/FeatureImportance'
import TransactionTable from './components/TransactionTable'
import RetrainPanel from './components/RetrainPanel'
import './App.css'

const API = '/api'

function App() {
  const [page, setPage] = useState('upload') // 'upload' | 'dashboard'
  const [overview, setOverview] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [driftScores, setDriftScores] = useState([])
  const [alerts, setAlerts] = useState([])
  const [fraudMix, setFraudMix] = useState([])
  const [featureImportance, setFeatureImportance] = useState([])
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const loadDashboard = () => {
    setLoading(true)
    Promise.all([
      fetch(`${API}/overview`).then(r => r.json()),
      fetch(`${API}/metrics`).then(r => r.json()),
      fetch(`${API}/drift-scores`).then(r => r.json()),
      fetch(`${API}/alerts`).then(r => r.json()),
      fetch(`${API}/fraud-type-mix`).then(r => r.json()),
      fetch(`${API}/feature-importance`).then(r => r.json()),
      fetch(`${API}/transactions?per_page=100`).then(r => r.json()),
    ])
      .then(([ov, met, drift, al, mix, fi, txn]) => {
        setOverview(ov)
        setMetrics(met)
        setDriftScores(drift)
        setAlerts(al)
        setFraudMix(mix)
        setFeatureImportance(fi)
        setTransactions(txn.transactions)
        setPage('dashboard')
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }

  if (page === 'upload') {
    return <UploadPage onReady={loadDashboard} />
  }

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        <p>Loading dashboard...</p>
      </div>
    )
  }

  if (error) {
    return <div className="error">Error: {error}</div>
  }

  return (
    <div className="app">
      <header>
        <div className="header-content">
          <h1>Fraud Detection Monitor</h1>
          <p className="subtitle">
            {overview?.data_source && (
              <span>Data: {overview.data_source} &middot; </span>
            )}
            Model performance tracking with concept drift detection
          </p>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {overview && overview.critical_alerts > 0 && (
            <div className="header-alert">
              {overview.critical_alerts} critical alert{overview.critical_alerts > 1 ? 's' : ''}
            </div>
          )}
          <button
            className="new-upload-btn"
            onClick={() => setPage('upload')}
          >
            New Upload
          </button>
        </div>
      </header>

      <main>
        <Overview data={overview} />

        <div className="grid-2">
          <MetricsChart data={metrics} baseline={overview?.baseline_auprc} />
          <DriftChart data={driftScores} />
        </div>

        <AlertsPanel alerts={alerts} />

        <RetrainPanel
          overview={overview}
          onRetrained={(result) => {
            setOverview(prev => ({
              ...prev,
              retrain_urgency: 'none',
              retrain_reasons: ['Model was just retrained'],
            }))
          }}
        />

        <div className="grid-2">
          <FraudMixChart data={fraudMix} />
          <FeatureImportance data={featureImportance} />
        </div>

        <TransactionTable transactions={transactions} />
      </main>

      <footer>
        <p>Simulated fraud detection pipeline with concept drift monitoring</p>
      </footer>
    </div>
  )
}

export default App
