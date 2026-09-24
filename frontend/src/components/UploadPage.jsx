import { useState, useRef } from 'react'

export default function UploadPage({ onReady }) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const fileInput = useRef(null)

  const runPipeline = async (mode, file) => {
    setLoading(true)
    setError(null)
    setProgress(mode === 'demo' ? 'Generating demo data...' : 'Uploading file...')

    try {
      let res
      if (mode === 'demo') {
        setProgress('Running simulation & training model (this takes ~60-90s)...')
        res = await fetch('/api/demo', { method: 'POST' })
      } else {
        setProgress('Processing CSV...')
        const formData = new FormData()
        formData.append('file', file)
        res = await fetch('/api/upload', { method: 'POST', body: formData })
      }

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Pipeline failed')
      }

      const result = await res.json()
      setUploadResult(result)
      setProgress('Loading dashboard...')
      onReady()
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  const handleFile = (file) => {
    if (!file) return
    if (!file.name.endsWith('.csv')) {
      setError('Please upload a CSV file')
      return
    }
    runPipeline('upload', file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  if (loading) {
    return (
      <div className="upload-page">
        <div className="upload-loading">
          <div className="spinner" />
          <p className="upload-progress-text">{progress}</p>
          <p className="upload-progress-sub">
            Training LightGBM + XGBoost ensemble, running drift detection...
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>Fraud Detection Monitor</h1>
          <p>Upload your transaction history to train a fraud detection model and monitor it for concept drift</p>
        </div>

        <div
          className={`upload-dropzone ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInput.current?.click()}
        >
          <input
            ref={fileInput}
            type="file"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          <div className="upload-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="upload-cta">Drop a CSV file here, or click to browse</p>
          <p className="upload-hint">Transaction data with at least <code>amount</code> and <code>is_fraud</code> columns</p>
        </div>

        {error && (
          <div className="upload-error">
            {error}
          </div>
        )}

        <div className="upload-divider">
          <span>or</span>
        </div>

        <button className="demo-btn" onClick={() => runPipeline('demo')}>
          Use Demo Data
          <span className="demo-btn-sub">150K simulated transactions with concept drift</span>
        </button>

        <div className="upload-format">
          <h3>Expected CSV format</h3>
          <div className="format-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Required</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>amount</code></td>
                  <td><span className="req-badge yes">Required</span></td>
                  <td>Transaction amount in dollars</td>
                </tr>
                <tr>
                  <td><code>is_fraud</code></td>
                  <td><span className="req-badge yes">Required</span></td>
                  <td>Fraud label (0 or 1)</td>
                </tr>
                <tr>
                  <td><code>timestamp</code></td>
                  <td><span className="req-badge rec">Recommended</span></td>
                  <td>Transaction date/time (for temporal split & drift)</td>
                </tr>
                <tr>
                  <td><code>customer_id</code></td>
                  <td><span className="req-badge opt">Optional</span></td>
                  <td>Customer identifier</td>
                </tr>
                <tr>
                  <td><code>merchant_category</code></td>
                  <td><span className="req-badge opt">Optional</span></td>
                  <td>Merchant type (e.g. grocery, electronics)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="format-note">
            Missing columns are auto-synthesized. Column names are matched flexibly
            (e.g. <code>amt</code>, <code>transaction_amount</code>, <code>trans_amt</code> all map to <code>amount</code>).
          </p>
        </div>
      </div>
    </div>
  )
}
