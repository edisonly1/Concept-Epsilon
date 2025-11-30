import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, ReferenceLine 
} from 'recharts';
import { 
  Activity, Droplet, AlertTriangle, CheckCircle, Play, Search, AlertOctagon, 
  Lock, Map as MapIcon, ChevronRight, LayoutDashboard, History, User, LogOut, ArrowRight 
} from 'lucide-react';
import './index.css';

const API_URL = "http://127.0.0.1:8000/api";

// --- API CLIENT ---

const simulateLeak = async (leakNode, coeff) => {
  const response = await fetch(`${API_URL}/events/predict_leak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      network_name: "Tnet1.inp",
      event_type: "valve_closure",
      element_id: "VALVE",
      leak_node_id: leakNode,
      emitter_coeff: coeff,
      duration_seconds: 20
    })
  });
  if (!response.ok) throw new Error("Simulation failed");
  return response.json();
};

// --- SUB-COMPONENTS ---

const NetworkMap = ({ activeLeak, detectedCandidates }) => {
  const nodes = {
    R1: { x: 50, y: 150, type: 'reservoir' },
    N3: { x: 150, y: 150, type: 'junction' },
    N4: { x: 250, y: 100, type: 'junction' },
    N2: { x: 250, y: 200, type: 'junction' },
    N6: { x: 350, y: 100, type: 'junction' },
    N5: { x: 350, y: 200, type: 'junction' },
    N7: { x: 450, y: 200, type: 'junction' },
    N8: { x: 550, y: 200, type: 'junction' },
  };

  const pipes = [
    { from: 'R1', to: 'N3' },
    { from: 'N3', to: 'N4' },
    { from: 'N3', to: 'N2' },
    { from: 'N4', to: 'N6' },
    { from: 'N2', to: 'N5' },
    { from: 'N4', to: 'N2' },
    { from: 'N6', to: 'N5' },
    { from: 'N5', to: 'N7' },
  ];

  const valves = [
    { from: 'N7', to: 'N8', id: 'VALVE' }
  ];

  const getNodeColor = (id) => {
    if (activeLeak === id) return "#ef4444";
    const candidate = detectedCandidates.find(c => c.pipe_id === id);
    if (candidate) {
        if (candidate.rank === 0) return "#f59e0b";
        return "#fcd34d";
    }
    return "#3b82f6";
  };

  const isZoneHighlighed = (n1, n2) => {
     if (detectedCandidates.length < 2) return false;
     const c1 = detectedCandidates[0];
     const c2 = detectedCandidates[1];
     const candidateIds = [c1.pipe_id, c2.pipe_id];
     return candidateIds.includes(n1) && candidateIds.includes(n2) && Math.abs(c1.physics_score - c2.physics_score) < 0.001;
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
      <h3 className="text-lg font-semibold text-slate-700 mb-4 flex items-center gap-2">
        <Activity size={20} /> Network Topology (Tnet1)
      </h3>
      <div className="flex-grow w-full bg-slate-50 rounded-lg flex items-center justify-center overflow-hidden border border-slate-100 relative">
        <div className="absolute top-2 left-2 text-xs text-slate-400 font-mono">scale: 1:1000</div>
        <svg viewBox="0 0 600 300" className="w-full h-full max-w-2xl">
          {pipes.map((p, i) => (
            <line key={i} x1={nodes[p.from].x} y1={nodes[p.from].y} x2={nodes[p.to].x} y2={nodes[p.to].y} stroke="#94a3b8" strokeWidth="3" />
          ))}
          {valves.map((v, i) => {
             const highlight = isZoneHighlighed(v.from, v.to);
             return (
              <g key={`v-${i}`}>
                <line x1={nodes[v.from].x} y1={nodes[v.from].y} x2={nodes[v.to].x} y2={nodes[v.to].y} stroke={highlight ? "#d97706" : "#475569"} strokeWidth={highlight ? "6" : "4"} strokeDasharray="4 2" />
                {highlight && (
                    <text x={(nodes[v.from].x + nodes[v.to].x)/2} y={nodes[v.from].y - 15} textAnchor="middle" fill="#d97706" fontSize="12" fontWeight="bold">Likely Leak Zone</text>
                )}
              </g>
            );
          })}
          {Object.entries(nodes).map(([id, pos]) => (
            <g key={id}>
              <circle cx={pos.x} cy={pos.y} r={activeLeak === id ? 12 : 8} fill={getNodeColor(id)} stroke="white" strokeWidth="2" className="transition-all duration-300" />
              <text x={pos.x} y={pos.y - 15} textAnchor="middle" className="text-xs font-bold fill-slate-600">{id}</text>
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
};

const HistorySidebar = ({ history, onLoadHistory }) => (
    <div className="w-72 bg-white border-l border-slate-200 p-4 h-full hidden xl:block overflow-y-auto">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
          <History size={14}/> Session Log
      </h3>
      <div className="space-y-3">
        {history.length === 0 && <div className="text-sm text-slate-400 italic text-center py-4">No simulations run yet.</div>}
        {history.map(entry => (
          <button 
              key={entry.id} 
              onClick={() => onLoadHistory(entry)}
              className="w-full text-left p-3 rounded-lg hover:bg-slate-50 border border-slate-100 hover:border-slate-300 transition-all text-sm group"
          >
            <div className="flex justify-between items-center mb-1">
                <span className="font-bold text-slate-700">Target: {entry.target}</span>
                <span className="text-[10px] text-slate-400">{entry.timestamp}</span>
            </div>
            <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${entry.result.leak_present ? 'bg-red-500' : 'bg-green-500'}`}></div>
                <span className={`text-xs font-medium ${entry.result.leak_present ? 'text-red-600' : 'text-green-600'}`}>
                    {entry.result.leak_present ? `Leak Detected (${(entry.result.leak_probability * 100).toFixed(0)}%)` : 'System Secure'}
                </span>
            </div>
          </button>
        ))}
      </div>
    </div>
);

const ForensicsChart = ({ candidates }) => {
    const data = candidates.map(c => ({
        name: c.pipe_id,
        AI: c.model_prob,
        Physics: c.physics_score,
        Combined: c.combined_score
    }));

    return (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
            <h3 className="text-lg font-semibold mb-4 text-slate-700 flex items-center gap-2">
                <Search size={20}/> AI vs. Physics Verification
            </h3>
            <div className="flex-grow w-full min-h-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical" margin={{ left: 20 }}>
                        <XAxis type="number" domain={[0, 1]} hide />
                        <YAxis dataKey="name" type="category" width={30} tick={{fontSize: 12, fontWeight: 'bold'}} />
                        <Tooltip 
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            cursor={{fill: '#f1f5f9'}}
                        />
                        <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}/>
                        <Bar name="AI Confidence" dataKey="AI" fill="#94a3b8" radius={[0, 4, 4, 0]} barSize={12} />
                        <Bar name="Hydraulic Physics" dataKey="Physics" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={12} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <p className="text-xs text-slate-400 mt-2 text-center">
                Discrepancies between AI and Physics scores indicate potential model hallucinations.
            </p>
        </div>
    );
};

const StatGrid = ({ result }) => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
        <div className="p-3 bg-purple-50 text-purple-600 rounded-lg">
            <AlertTriangle size={24} />
        </div>
        <div>
            <div className="text-slate-500 text-xs font-bold uppercase tracking-wider">Model Uncertainty</div>
            <div className="text-2xl font-bold text-slate-800">
                {result ? (result.uncertainty * 100).toFixed(1) : 0}<span className="text-sm text-slate-400 font-normal">%</span>
            </div>
        </div>
      </div>
      
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
        <div className="p-3 bg-blue-50 text-blue-600 rounded-lg">
            <Activity size={24} />
        </div>
        <div>
            <div className="text-slate-500 text-xs font-bold uppercase tracking-wider">System Pressure (Avg)</div>
            <div className="text-2xl font-bold text-slate-800">
                {result && result.observed_data.length > 0 
                    ? (result.observed_data[0].head.values.reduce((a,b)=>a+b,0)/result.observed_data[0].head.values.length).toFixed(0) 
                    : 120} <span className="text-sm text-slate-400 font-normal">PSI</span>
            </div>
        </div>
      </div>
      
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center gap-4">
        <div className={`p-3 rounded-lg ${result?.leak_probability > 0.5 ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'}`}>
            <Droplet size={24} />
        </div>
        <div>
            <div className="text-slate-500 text-xs font-bold uppercase tracking-wider">Leak Probability</div>
            <div className={`text-2xl font-bold ${result?.leak_probability > 0.5 ? 'text-red-600' : 'text-slate-800'}`}>
                {result ? (result.leak_probability * 100).toFixed(1) : 0}<span className="text-sm text-slate-400 font-normal">%</span>
            </div>
        </div>
      </div>
    </div>
  );

// --- MAIN PAGES ---

const LoginPage = ({ onLogin }) => {
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        setIsLoading(true);
        // Simulate auth delay
        setTimeout(() => {
            setIsLoading(false);
            onLogin();
        }, 800);
    };

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
            <div className="bg-white p-8 rounded-2xl shadow-xl border border-slate-200 max-w-md w-full animate-in slide-in-from-bottom-4 duration-700">
                <div className="mb-8 text-center">
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-600 rounded-xl mb-4 shadow-lg shadow-blue-600/20">
                        <Droplet className="text-white w-6 h-6" />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-800">Welcome Back</h2>
                    <p className="text-slate-500 text-sm mt-1">Sign in to Epsilon Engineer Portal</p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-1">
                        <label className="block text-xs font-bold text-slate-500 uppercase">Email</label>
                        <input type="email" defaultValue="demo@epsilon.com" className="w-full p-3 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:ring-2 focus:ring-blue-500 outline-none transition-all focus:border-blue-500" />
                    </div>
                    <div className="space-y-1">
                        <label className="block text-xs font-bold text-slate-500 uppercase">Password</label>
                        <input type="password" defaultValue="password" className="w-full p-3 bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:ring-2 focus:ring-blue-500 outline-none transition-all focus:border-blue-500" />
                    </div>
                    <button 
                        type="submit" 
                        disabled={isLoading}
                        className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2 mt-4"
                    >
                        {isLoading ? (
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"/>
                        ) : (
                            <>Sign In <ArrowRight size={18} /></>
                        )}
                    </button>
                </form>
                
                <div className="mt-8 text-center text-xs text-slate-400">
                    &copy; 2025 Epsilon Systems. Secure Access Only.
                </div>
            </div>
        </div>
    );
};

const NetworkSelectionPage = ({ onSelect, onLogout }) => {
    const networks = [
        { id: 'tnet1', name: 'Tnet1 Demo System', status: 'active', desc: '8 Nodes, 1 Reservoir' },
        { id: 'austin', name: 'Austin District 4', status: 'locked', desc: 'Unavailable in Demo' },
        { id: 'kyoto', name: 'Kyoto Water Grid', status: 'locked', desc: 'Unavailable in Demo' },
    ];

    return (
        <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4 animate-in fade-in duration-500">
            <div className="max-w-md w-full space-y-6">
                <div className="text-center space-y-2">
                    <div className="inline-flex p-3 bg-white text-blue-600 rounded-xl mb-2 shadow-sm border border-slate-100">
                        <MapIcon size={24} />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-800">Select Environment</h2>
                    <p className="text-slate-500 text-sm">Choose a network topology to load.</p>
                </div>

                <div className="bg-white p-2 rounded-2xl shadow-sm border border-slate-200 space-y-2">
                    {networks.map(net => (
                        <button 
                            key={net.id}
                            disabled={net.status === 'locked'}
                            onClick={() => onSelect(net.id)}
                            className={`w-full p-4 rounded-xl border text-left transition-all relative group flex items-center justify-between
                                ${net.status === 'active' 
                                    ? 'bg-white border-transparent hover:bg-blue-50 hover:border-blue-200 cursor-pointer shadow-sm hover:shadow' 
                                    : 'bg-slate-50 border-transparent opacity-50 cursor-not-allowed'}`}
                        >
                            <div className="flex items-center gap-4">
                                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${net.status === 'active' ? 'bg-blue-100 text-blue-600' : 'bg-slate-200 text-slate-400'}`}>
                                    <Activity size={20} />
                                </div>
                                <div>
                                    <div className="font-semibold text-slate-800">{net.name}</div>
                                    <div className="text-xs text-slate-500">{net.desc}</div>
                                </div>
                            </div>
                            {net.status === 'locked' && <Lock size={16} className="text-slate-300" />}
                            {net.status === 'active' && <ChevronRight size={20} className="text-blue-400" />}
                        </button>
                    ))}
                </div>

                <div className="text-center">
                    <button onClick={onLogout} className="text-sm text-slate-400 hover:text-red-500 font-medium transition-colors flex items-center justify-center gap-2 mx-auto">
                        <LogOut size={14} /> Sign out
                    </button>
                </div>
            </div>
        </div>
    );
};

const Dashboard = ({ onLogout }) => {
  const [loading, setLoading] = useState(false);
  const [simulatedLeakNode, setSimulatedLeakNode] = useState("N7");
  const [result, setResult] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [history, setHistory] = useState([]);

  const handleSimulate = async () => {
    setLoading(true);
    setResult(null);
    try {
      const node = simulatedLeakNode === "None" ? null : simulatedLeakNode;
      const res = await simulateLeak(node, 0.01);
      
      // Update Chart Data with Real Physics
      if (res.observed_data && res.observed_data.length > 0) {
        const sensor = res.observed_data[0];
        const realData = sensor.head.t.map((time, index) => ({
          time: time.toFixed(2),
          pressure: sensor.head.values[index]
        }));
        setChartData(realData);
      } else {
        setChartData([]);
      }
      
      setResult(res);

      // Add to History
      const newEntry = { 
        id: Date.now(), 
        timestamp: new Date().toLocaleTimeString(), 
        target: node || "None", 
        result: res,
        data: res.observed_data // store data for history reload
      };
      setHistory(prev => [newEntry, ...prev]);

    } catch (err) {
      console.error(err);
      alert("Simulation failed. Check Backend!");
    } finally {
      setLoading(false);
    }
  };

  const loadHistoryEntry = (entry) => {
      setResult(entry.result);
      if (entry.data && entry.data.length > 0) {
        const sensor = entry.data[0];
        const realData = sensor.head.t.map((time, index) => ({
            time: time.toFixed(2),
            pressure: sensor.head.values[index]
        }));
        setChartData(realData);
      }
      setSimulatedLeakNode(entry.target === "None" ? "N5" : entry.target); // simplified reset
  };

  const getDiagnosis = () => {
    if (!result) return null;
    if (!result.leak_present) return { title: "System Secure", desc: "No leaks detected.", color: "text-green-600", icon: CheckCircle };

    const top = result.top_candidates;
    if (top.length >= 2) {
        const scoreDiff = Math.abs(top[0].physics_score - top[1].physics_score);
        if (scoreDiff < 0.001) {
            return {
                title: "Ambiguous Localization",
                desc: `Signal matches both ${top[0].pipe_id} and ${top[1].pipe_id}. The leak is likely located at the component connecting them (VALVE).`,
                color: "text-amber-600",
                icon: AlertOctagon,
                isAmbiguous: true
            };
        }
    }
    return {
        title: `Leak Detected at ${top[0].pipe_id}`,
        desc: `Confidence: ${(top[0].combined_score * 100).toFixed(1)}%`,
        color: "text-red-600",
        icon: AlertTriangle
    };
  };

  const diagnosis = getDiagnosis();

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden animate-in fade-in duration-500">
        
        {/* Main Content Scrollable Area */}
        <div className="flex-1 flex flex-col h-full overflow-hidden">
            
            {/* Top Navigation */}
            <header className="bg-white border-b border-slate-200 px-6 py-3 flex justify-between items-center shrink-0">
                <div className="flex items-center gap-3">
                    <Droplet className="text-blue-500 fill-blue-500" /> 
                    <h1 className="text-xl font-bold text-slate-800">Epsilon <span className="text-slate-400 font-normal">/ Tnet1 Workspace</span></h1>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-full">
                        <User size={16} className="text-slate-500"/>
                        <span className="text-sm font-medium text-slate-600">Demo User</span>
                    </div>
                    <button onClick={onLogout} className="text-slate-400 hover:text-red-500 transition-colors">
                        <LogOut size={20} />
                    </button>
                </div>
            </header>

            {/* Dashboard Content */}
            <main className="flex-1 overflow-y-auto p-6">
                <div className="max-w-7xl mx-auto">
                    
                    {/* Top Stats Row */}
                    <StatGrid result={result} />

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6 h-auto min-h-[400px]">
                        {/* Control Panel */}
                        <div className="space-y-6 flex flex-col">
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><Play size={20} /> Simulation Control</h3>
                                <div className="space-y-4">
                                    <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1">Simulate Leak At</label>
                                    <select value={simulatedLeakNode} onChange={(e) => setSimulatedLeakNode(e.target.value)} className="w-full p-2 border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none">
                                        <option value="N5">Node N5</option>
                                        <option value="N6">Node N6</option>
                                        <option value="N7">Node N7</option>
                                        <option value="N8">Node N8</option>
                                    </select>
                                    </div>
                                    <button onClick={handleSimulate} disabled={loading} className={`w-full py-3 px-4 rounded-lg font-medium text-white flex items-center justify-center gap-2 transition-colors ${loading ? 'bg-slate-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg'}`}>
                                    {loading ? 'Processing Physics...' : 'Run Analysis'}
                                    </button>
                                </div>
                            </div>

                            {result && diagnosis && (
                                <div className={`flex-1 bg-white p-6 rounded-xl shadow-sm border-l-4 ${diagnosis.color === 'text-red-600' ? 'border-red-500' : diagnosis.color === 'text-amber-600' ? 'border-amber-500' : 'border-green-500'}`}>
                                    <div className="flex items-start gap-4">
                                    <div className={`p-3 rounded-full bg-slate-100 ${diagnosis.color}`}><diagnosis.icon size={28} /></div>
                                    <div>
                                        <h4 className={`text-lg font-bold ${diagnosis.color}`}>{diagnosis.title}</h4>
                                        <p className="text-slate-600 mt-1 text-sm leading-relaxed">{diagnosis.desc}</p>
                                        {diagnosis.isAmbiguous && <div className="mt-3 bg-amber-50 p-3 rounded text-xs text-amber-800 border border-amber-100"><strong>Engineering Note:</strong> N7 and N8 have identical physics scores. Inspect the Valve.</div>}
                                    </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Network Map (Takes up 2 cols) */}
                        <div className="lg:col-span-2">
                             <NetworkMap activeLeak={simulatedLeakNode} detectedCandidates={result?.top_candidates?.map((c, i) => ({...c, rank: i})) || []} />
                        </div>
                    </div>

                    {/* Bottom Row: Charts */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[350px]">
                        {/* Real-time Sensor Data */}
                        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex flex-col">
                             <h3 className="text-lg font-semibold mb-4 text-slate-700 flex items-center gap-2">
                                <LayoutDashboard size={20}/> Sensor Readings (N4)
                             </h3>
                             <div className="flex-grow w-full">
                                {chartData.length > 0 ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={chartData}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                            <XAxis dataKey="time" hide />
                                            <YAxis domain={['auto', 'auto']} width={40} tick={{fontSize: 12}} />
                                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }} />
                                            <Line type="monotone" dataKey="pressure" stroke="#3b82f6" strokeWidth={3} dot={false} animationDuration={500} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                ) : <div className="h-full flex items-center justify-center text-slate-400 italic">Run a simulation to view live sensor data</div>}
                             </div>
                        </div>

                        {/* Forensics Chart */}
                        {result && result.leak_present ? (
                            <ForensicsChart candidates={result.top_candidates} />
                        ) : (
                            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full flex items-center justify-center text-slate-400 italic border-dashed">
                                No leak detected. Forensics module inactive.
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>

        {/* Right Sidebar: History */}
        <HistorySidebar history={history} onLoadHistory={loadHistoryEntry} />
    </div>
  );
};

function App() {
  const [authStep, setAuthStep] = useState('login'); // 'login' | 'network' | 'dashboard'
  const [activeNetwork, setActiveNetwork] = useState(null);

  const handleLogin = () => {
      setAuthStep('network');
  };

  const handleNetworkSelect = (networkId) => {
      setActiveNetwork(networkId);
      setAuthStep('dashboard');
  };

  const handleLogout = () => {
      setAuthStep('login');
      setActiveNetwork(null);
  };

  return (
    <>
      {authStep === 'login' && <LoginPage onLogin={handleLogin} />}
      {authStep === 'network' && <NetworkSelectionPage onSelect={handleNetworkSelect} onLogout={handleLogout} />}
      {authStep === 'dashboard' && <Dashboard onLogout={handleLogout} />}
    </>
  );
}

const rootElement = document.getElementById('root');
if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(<App />);
}
