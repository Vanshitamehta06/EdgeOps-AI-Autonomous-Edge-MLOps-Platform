import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const GRAFANA_PREVIEW_URL = 'http://localhost:3000/d-solo/adgvq6f/edgeops-live?orgId=1&from=now-30m&to=now&timezone=browser&panelId=panel-1&theme=light&kiosk';
const GRAFANA_ML_HEALTH_URL = 'http://localhost:3000/d-solo/adwk9d5/ml-health-edgeops-live?orgId=1&from=now-30m&to=now&timezone=browser&panelId=panel-1&theme=light&kiosk';
const GRAFANA_FULL_URL = 'http://localhost:3000/d/adgvq6f/edgeops-live?orgId=1&refresh=5s';

const MetricBox = ({ title, value }) => (
  <div className="hover-card" style={{ background: 'white', padding: '20px', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)' }}>
    <div style={{ color: '#64748b', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 'bold', marginBottom: '8px' }}>{title}</div>
    <div style={{ color: '#0f172a', fontSize: '1.8rem', fontWeight: '900', background: 'linear-gradient(to right, #0f172a, #334155)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{value}</div>
  </div>
);

function App() {
  const [apiStatus, setApiStatus] = useState(false);
  const [registry, setRegistry] = useState([]);
  const [deployState, setDeployState] = useState({});
  const [driftState, setDriftState] = useState({ drift_detected: false, drift_share: 0.0 });
  const [isLoading, setIsLoading] = useState(false);


  const handleGrafanaClick = () => {
    window.open(GRAFANA_FULL_URL, '_blank', 'noopener,noreferrer');
  };

  const fetchData = async () => {
    try {
      const healthRes = await axios.get(`${API_BASE_URL}/health`, { timeout: 2000 });
      setApiStatus(healthRes.status === 200);

      const registryRes = await axios.get(`${API_BASE_URL}/dashboard/registry`);
      setRegistry(Array.isArray(registryRes.data) ? registryRes.data : []);

      const stateRes = await axios.get(`${API_BASE_URL}/dashboard/state`);
      setDeployState(stateRes.data || {});

      const driftRes = await axios.get(`${API_BASE_URL}/dashboard/drift`);
      setDriftState(driftRes.data || { drift_detected: false, drift_share: 0.0 });
    } catch (error) {
      setApiStatus(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleAction = async (actionName, endpoint) => {
    setIsLoading(true);
    try {
        await axios.post(`${API_BASE_URL}/${endpoint}`);
        alert(`${actionName} executed successfully!`);
    } catch(e) {
        const backendError = e.response?.data?.detail || e.message;
        alert(`${actionName} failed:\n\n${backendError}`);
        console.error("Full error details:", e);
    }
    fetchData();
    setIsLoading(false);
  };

  const activeModel = registry.find(m => m.id === deployState.current_model_id);

  return (
    <>
      <style>{`
        body { margin: 0; background-color: #f1f5f9; color: #0f172a; }
        .hover-card { transition: transform 0.2s ease, box-shadow 0.2s ease; }
        .hover-card:hover { transform: translateY(-4px); box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.1); }
        .btn { transition: all 0.2s ease; outline: none; }
        .btn:hover:not(:disabled) { transform: scale(1.02); filter: brightness(1.1); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .btn:active:not(:disabled) { transform: scale(0.98); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .chart-bar { transition: height 0.8s cubic-bezier(0.4, 0, 0.2, 1), filter 0.2s; }
        .chart-bar:hover { filter: brightness(1.2); }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
          100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
        }
        .pulse-dot { width: 10px; height: 10px; background-color: #22c55e; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
        .pulse-dot-red { background-color: #ef4444; animation: none; box-shadow: 0 0 5px #ef4444; }
        .custom-scroll::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scroll::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
      `}</style>
      
      <div style={{ display: 'flex', minHeight: '100vh', fontFamily: '"Inter", "Segoe UI", sans-serif' }}>
        
        {/* STICKY SIDEBAR */}
        <div style={{ 
          width: '260px', 
          background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)', 
          color: 'white', 
          padding: '25px', 
          display: 'flex', 
          flexDirection: 'column', 
          boxShadow: '4px 0 15px rgba(0,0,0,0.1)', 
          zIndex: 10,
          position: 'sticky', 
          top: 0,             
          height: '100vh',    
          boxSizing: 'border-box'
        }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginTop: 0, fontSize: '1.4rem' }}>
            <span style={{ background: '#3b82f6', padding: '8px', borderRadius: '8px' }}>⚡</span> EdgeOps AI
          </h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '-10px', marginBottom: '30px' }}>Automated MLOps Platform</p>
          
          <div style={{ flex: 1, overflowY: 'auto' }}> 
            <h4 style={{ margin: '10px 0', color: '#cbd5e1', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Pipeline Controls</h4>
            <button className="btn" onClick={() => handleAction('Pipeline', 'run_pipeline')} disabled={isLoading} style={{ width: '100%', padding: '12px', background: '#334155', color: 'white', border: '1px solid #475569', borderLeft: '4px solid #eab308', borderRadius: '8px', cursor: 'pointer', marginBottom: '16px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
              ⚙️ Run Initial Pipeline
            </button>
            <button className="btn" onClick={() => handleAction('Stream', 'simulate_stream')} disabled={isLoading} style={{ width: '100%', padding: '12px', background: 'linear-gradient(135deg, #2563eb, #3b82f6)', color: 'white', border: 'none', borderLeft: '4px solid #eab308', borderRadius: '8px', cursor: 'pointer', marginBottom: '30px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
              🌊 Simulate Live Stream
            </button>

            <h4 style={{ margin: '10px 0', color: '#cbd5e1', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Maintenance</h4>
            <button className="btn" onClick={() => handleAction('Optimize', 'force_optimization')} disabled={isLoading} style={{ width: '100%', padding: '12px', background: '#334155', color: 'white', border: '1px solid #475569', borderLeft: '4px solid #eab308', borderRadius: '8px', cursor: 'pointer', marginBottom: '16px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
              🔄 Force Optimization
            </button>
            <button className="btn" onClick={() => handleAction('Hard Reset', 'hard_reset')} disabled={isLoading} style={{ width: '100%', padding: '12px', background: '#334155', color: 'white', border: '1px solid #475569', borderLeft: '4px solid #eab308', borderRadius: '8px', cursor: 'pointer', marginBottom: '16px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
              🛠️ Log Hardware Reset
            </button>
          </div>
          
          <div style={{ marginTop: 'auto', paddingTop: '20px' }}> 
            <h4 style={{ margin: '10px 0', color: '#cbd5e1', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Emergency</h4>
            <button className="btn" onClick={() => handleAction('Rollback', 'rollback')} disabled={isLoading} style={{ width: '100%', padding: '12px', background: 'linear-gradient(135deg, #dc2626, #ef4444)', color: 'white', border: 'none', borderLeft: '4px solid #eab308', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '10px' }}>
              ⏪ Rollback Deployment
            </button>
          </div>
        </div>

        {/* MAIN CONTENT */}
        <div className="custom-scroll" style={{ flex: 1, padding: '40px', overflowY: 'auto', maxWidth: '1400px', margin: '0 auto' }}>
          
          {/* HEADER */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '30px' }}>
            <div>
              <h1 style={{ margin: 0, fontSize: '2.2rem', color: '#0f172a', letterSpacing: '-0.02em' }}>🏭 Predictive Maintenance</h1>
              <p style={{ color: '#64748b', marginTop: '8px', fontSize: '1.1rem' }}>Live Edge Monitoring & Automated Optimization Control Center.</p>
            </div>
            <div className="hover-card" style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 20px', borderRadius: '50px', background: 'white', border: '1px solid #e2e8f0', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <span className={apiStatus ? "pulse-dot" : "pulse-dot pulse-dot-red"}></span>
              <span style={{ fontWeight: 'bold', color: apiStatus ? '#166534' : '#991b1b' }}>
                API {apiStatus ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>

          {/* DRIFT ALARM */}
          <div style={{ transition: 'all 0.3s ease' }}>
            {(driftState.drift_detected || driftState.drift_share >= 0.3) ? (
              <div className="hover-card" style={{ background: 'linear-gradient(to right, #fef2f2, #fee2e2)', borderLeft: '6px solid #ef4444', color: '#991b1b', padding: '20px', borderRadius: '8px', marginBottom: '30px', display: 'flex', gap: '15px', alignItems: 'center', boxShadow: '0 4px 6px rgba(239,68,68,0.1)' }}>
                <span style={{ fontSize: '1.5rem' }}>🚨</span>
                <div>
                  <h4 style={{ margin: 0, fontSize: '1.1rem' }}>SYSTEM ALARM: Data Drift Detected!</h4>
                  <p style={{ margin: '5px 0 0 0', opacity: 0.9 }}>{(driftState.drift_share * 100).toFixed(1)}% of features have drifted. The optimizer is spinning up a new model.</p>
                </div>
              </div>
            ) : (
              <div className="hover-card" style={{ background: 'white', borderLeft: '6px solid #22c55e', color: '#166534', padding: '20px', borderRadius: '8px', marginBottom: '30px', display: 'flex', gap: '15px', alignItems: 'center' }}>
                <span style={{ fontSize: '1.5rem' }}>🛡️</span>
                <div>
                  <h4 style={{ margin: 0, fontSize: '1.1rem' }}>System Healthy</h4>
                  <p style={{ margin: '5px 0 0 0', opacity: 0.8 }}>No significant drift detected in recent data logs ({(driftState.drift_share * 100).toFixed(1)}% shift).</p>
                </div>
              </div>
            )}
          </div>
          

          {/* METRICS GRID */}
          <h3 style={{ marginTop: 0, display: 'flex', gap: '10px', alignItems: 'center', color: '#334155' }}>💾 Active Edge Deployment</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '40px' }}>
            {activeModel ? (
              <>
                <MetricBox title="Active Version" value={`v${activeModel.version_number}`} />
                <MetricBox title="ROC-AUC" value={(activeModel.roc_auc || 0).toFixed(4)} />
                <MetricBox title="F1 Score" value={(activeModel.f1_score || 0).toFixed(4)} />
                <MetricBox title="Accuracy" value={(activeModel.accuracy || 0).toFixed(4)} />
                <MetricBox title="Fallback Target" value={`v${deployState.stable_model_id}`} />
              </>
            ) : (
               <div style={{ gridColumn: '1 / -1', padding: '30px', textAlign: 'center', background: 'white', borderRadius: '12px', color: '#64748b' }}>No active deployment found. Run the pipeline to generate models.</div>
            )}
          </div>

          {/* BOTTOM PANELS */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '30px' }}>
            
            {/* CHART PANEL */}
            <div className="hover-card" style={{ background: 'white', padding: '25px', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
               <h3 style={{ margin: '0 0 20px 0', color: '#334155' }}>📊 Optimization History (ROC-AUC)</h3>
               <div className="custom-scroll" style={{ display: 'flex', alignItems: 'flex-end', height: '250px', gap: '25px', padding: '20px 10px', paddingBottom: '15px', borderBottom: '2px solid #f1f5f9', overflowX: 'auto', overflowY: 'hidden' }}>
                  {registry.length === 0 ? (
                      <div style={{ width: '100%', textAlign: 'center', color: '#94a3b8' }}>Awaiting pipeline data...</div>
                  ) : (
                      [...registry].reverse().map((m) => {
                          const score = m.roc_auc || 0;
                          const heightPct = Math.max(10, (score - 0.5) * 200); 
                          return (
                              <div key={m.id} style={{ flex: '0 0 60px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px', height: '100%', justifyContent: 'flex-end' }}>
                                  <span style={{ fontSize: '0.85rem', color: '#64748b', fontWeight: 'bold' }}>{score.toFixed(3)}</span>
                                  <div className="chart-bar" style={{ width: '100%', maxWidth: '60px', background: 'linear-gradient(180deg, #3b82f6 0%, #2563eb 100%)', height: `${Math.min(100, heightPct)}%`, borderRadius: '6px 6px 0 0', boxShadow: '0 -2px 10px rgba(59, 130, 246, 0.2)' }}></div>
                                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold', color: '#0f172a' }}>v{m.version_number}</span>
                              </div>
                          );
                      })
                  )}
               </div>
            </div>

            {/* TABLE PANEL */}
            <div className="hover-card custom-scroll" style={{ background: 'white', padding: '25px', borderRadius: '12px', border: '1px solid #e2e8f0', overflowY: 'auto', maxHeight: '350px' }}>
              <h3 style={{ margin: '0 0 20px 0', color: '#334155' }}>📚 Model Artifact Registry</h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.95rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', color: '#64748b', textTransform: 'uppercase', fontSize: '0.8rem', letterSpacing: '0.05em' }}>
                    <th style={{ padding: '12px 10px', borderBottom: '2px solid #e2e8f0' }}>Version</th>
                    <th style={{ padding: '12px 10px', borderBottom: '2px solid #e2e8f0' }}>ROC-AUC</th>
                    <th style={{ padding: '12px 10px', borderBottom: '2px solid #e2e8f0' }}>F1 Score</th>
                    <th style={{ padding: '12px 10px', borderBottom: '2px solid #e2e8f0' }}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {registry.map(row => {
                    const isActive = row.id === deployState.current_model_id;
                    return (
                      <tr key={row.id} style={{ borderBottom: '1px solid #f1f5f9', background: isActive ? '#f0fdf4' : 'transparent', transition: 'background 0.2s' }}>
                        <td style={{ padding: '15px 10px', fontWeight: 'bold', color: isActive ? '#166534' : '#0f172a' }}>
                          v{row.version_number} {isActive && '⭐'}
                        </td>
                        <td style={{ padding: '15px 10px' }}>{(row.roc_auc || 0).toFixed(4)}</td>
                        <td style={{ padding: '15px 10px' }}>{(row.f1_score || 0).toFixed(4)}</td>
                        <td style={{ padding: '15px 10px', color: '#64748b' }}>{new Date(row.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* GRAFANA LIVE TELEMETRY PREVIEW */}
            <div 
              className="hover-card" 
              onClick={handleGrafanaClick}
              style={{ 
                background: 'white', 
                padding: '25px', 
                borderRadius: '12px', 
                border: '1px solid #e2e8f0', 
                position: 'relative', 
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                minHeight: '350px'
              }}
            >
              <h3 style={{ margin: '0 0 20px 0', color: '#334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>📈 Live Telemetry Stream</span>
                <span style={{ fontSize: '0.8rem', color: '#2563eb', fontWeight: '600', background: '#eff6ff', padding: '4px 10px', borderRadius: '20px' }}>View Full Panel ↗</span>
              </h3>
              
              <div style={{ position: 'relative', flex: 1, width: '100%', height: '100%' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }}></div>
                <iframe 
                  src={GRAFANA_PREVIEW_URL} 
                  width="100%" 
                  height="100%" 
                  frameBorder="0"
                  title="Grafana Preview Visual"
                  style={{ borderRadius: '8px', background: '#f8fafc' }}
                ></iframe>
              </div>
            </div>

            {/* ML HEALTH PANEL */}
            <div 
              className="hover-card" 
              onClick={() => window.open('http://localhost:3000/d/adwk9d5/ml-health-edgeops-live?orgId=1&refresh=5s', '_blank', 'noopener,noreferrer')}
              style={{ 
                background: 'white', 
                padding: '25px', 
                borderRadius: '12px', 
                border: '1px solid #e2e8f0', 
                position: 'relative', 
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                minHeight: '350px'
              }}
            >
              <h3 style={{ margin: '0 0 20px 0', color: '#334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>🤖 AI Pipeline Health</span>
                <span style={{ fontSize: '0.8rem', color: '#2563eb', fontWeight: '600', background: '#eff6ff', padding: '4px 10px', borderRadius: '20px' }}>View Full Panel ↗</span>
              </h3>
  
              <div style={{ position: 'relative', flex: 1, width: '100%', height: '100%' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', zIndex: 10 }}></div>
                <iframe 
                  src={GRAFANA_ML_HEALTH_URL} 
                  width="100%" 
                  height="100%" 
                  frameBorder="0"
                  title="ML Pipeline Health"
                  style={{ borderRadius: '8px', background: '#f8fafc' }}
                ></iframe>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
}

export default App;