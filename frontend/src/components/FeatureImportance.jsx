export default function FeatureImportance({ data }) {
  const maxVal = data.length > 0 ? data[0].importance : 1

  return (
    <div className="card">
      <h2>Feature Importance (Ensemble)</h2>
      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {data.map((item, i) => (
          <div key={i} className="fi-bar-container">
            <div className="fi-label">
              <span>{item.feature}</span>
              <span>{item.importance.toFixed(3)}</span>
            </div>
            <div className="fi-bar">
              <div
                className="fi-fill"
                style={{
                  width: `${(item.importance / maxVal) * 100}%`,
                  opacity: 0.5 + (item.importance / maxVal) * 0.5,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
