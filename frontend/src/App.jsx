import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { fetchModels, runAttack, fetchTree } from "./api.js";

const defenseInfo = {
  full: "The server returns full confidence values. This allows attackers to solve the equations exactly.",
  rounded: "The server rounds confidence values to two decimal places. The attack still works with a tiny error.",
  label_only: "The server only returns the final label. The attacker must fall back to slower retraining.",
  noisy: "The server adds noise. This drops the fidelity and raises the attack cost."
};

const modelNames = {
  fraud_lr: "Fraud Lr",
  multi_lr: "Multi-class Softmax",
  mlp: "Mlp",
  tree: "Decision Tree"
};

const defenseNames = {
  full: "Full",
  rounded: "Rounded",
  label_only: "Label Only",
  noisy: "Noisy"
};

export default function App() {
  const [models, setModels] = useState({});
  const [modelId, setModelId] = useState("fraud_lr");
  const [defense, setDefense] = useState("full");
  const [budget, setBudget] = useState(800);
  const [tab, setTab] = useState("lab");
  const [result, setResult] = useState(null);
  const [tree, setTree] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { fetchModels().then((responseData) => { setModels(responseData.models); }).catch(() => setError("Backend not reachable. Start uvicorn on port 8000.")); }, []);

  async function onRun() {
    setLoading(true); setError("");
    try {
      const attackResponse = await runAttack(modelId, defense, Number(budget));
      setResult(attackResponse);
      if (models[modelId]?.kind === "tree") {
        const treeResponse = await fetchTree(modelId);
        setTree(treeResponse.tree || null);
      }
    } catch {
      setError("Attack request failed. Check backend logs.");
    } finally { setLoading(false); }
  }

  const chartData = result?.curve?.length ? result.curve.map((curvePoint) => ({ queries: curvePoint.queries, fit: curvePoint.train_fit })) :
    result ? [{ queries: result.queries, fit: result.fidelity }] : [];

  return (
    <div className="layout">
      <aside>
        <h1>Model Extraction Lab</h1>
        <p style={{ color: "#9aa8bb" }}>Reproduction of Tramer et al. 2016. Steal a model using its prediction Api.</p>
        <label htmlFor="model">Victim Model</label>
        <select id="model" value={modelId} onChange={(e) => setModelId(e.target.value)}>
          {Object.entries(models).map(([id, modelInfo]) => <option key={id} value={id}>{modelInfo.desc}</option>)}
        </select>
        <label htmlFor="defense">Defense</label>
        <select id="defense" value={defense} onChange={(e) => setDefense(e.target.value)}>
          {["full", "rounded", "label_only", "noisy"].map((defenseOption) => <option key={defenseOption} value={defenseOption}>{defenseNames[defenseOption]}</option>)}
        </select>
        <p style={{ color: "#9aa8bb", fontSize: 13 }}>{defenseInfo[defense]}</p>
        <label htmlFor="budget">Query Budget</label>
        <input id="budget" type="number" min={9} max={4000} value={budget} onChange={(e) => setBudget(e.target.value)} />
        <button onClick={onRun} disabled={loading}>{loading ? "Attacking…" : "Run Extraction"}</button>
        {error && <div className="card" role="alert">{error}</div>}
      </aside>
      <main>
        <div className="tabs" role="tablist">
          {[["lab", "Lab"], ["tree", "Tree"], ["learn", "Learn"]].map(([tabKey, tabLabel]) => (
            <button key={tabKey} role="tab" aria-selected={tab === tabKey} className={tab === tabKey ? "active" : ""} onClick={() => setTab(tabKey)}>{tabLabel}</button>
          ))}
        </div>
        {tab === "lab" && (
          <>
            {!result && <div className="card">Pick a victim on the left and press Run Extraction. Binary Lr requires just 9 queries to steal the 8 feature model.</div>}
            {result && (
              <>
                <div className="kpis">
                  <div className="card"><div>Fidelity</div><div className="kpi mono">{(result.fidelity * 100).toFixed(2)}%</div></div>
                  <div className="card"><div>Queries</div><div className="kpi mono">{result.queries_used_this_attack}</div></div>
                  <div className="card"><div>Attack</div><div className="kpi" style={{ fontSize: 18 }}>{result.attack}</div></div>
                </div>
                <div className="card">
                  <h3>Fidelity vs Queries</h3>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={chartData}>
                      <CartesianGrid stroke="#374151" strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="queries" stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} axisLine={false} tickLine={false} />
                      <YAxis domain={[0.8, 1]} stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px', color: '#f9fafb'}} />
                      <Line type="monotone" dataKey="fit" stroke="#0891b2" strokeWidth={3} dot={false} activeDot={{r: 6, fill: '#0891b2', stroke: '#030712', strokeWidth: 2}} />
                    </LineChart>
                  </ResponsiveContainer>
                  <table aria-label="Fidelity numbers"><thead><tr><th>Queries</th><th>Score</th></tr></thead>
                    <tbody>{chartData.map((curvePoint, i) => <tr key={i}><td className="mono">{curvePoint.queries}</td><td className="mono">{Number(curvePoint.fit).toFixed(4)}</td></tr>)}</tbody></table>
                </div>
                <div className="card"><h3>Why This Works</h3><p>{result.note}</p>
                  {result.extracted && <pre className="mono">{JSON.stringify({ extracted: result.extracted, victim: result.victim, w_error: result.w_error }, null, 2)}</pre>}
                  {result.thresholds && <pre className="mono">{JSON.stringify(result.thresholds, null, 2)}</pre>}
                </div>
              </>
            )}
          </>
        )}
        {tab === "tree" && (
          <div className="card"><h3>Victim Tree Structure</h3>
            {!tree ? <p>Run the tree model to load its structure.</p> : <pre className="mono">{JSON.stringify(tree, null, 2)}</pre>}</div>
        )}
        {tab === "learn" && (
          <div className="card">
            <h3>How Model Extraction Works</h3>
            <p>Stealing a machine learning model is mostly basic algebra.</p>
            
            <h4 style={{ marginTop: '20px', marginBottom: '8px', color: 'var(--text)' }}>1. Linear And Logistic Regression</h4>
            <p>These models output exact confidence scores. An example is being 85 percent sure a transaction is fraud. The math behind them is a simple equation. An attacker sends a few random inputs and gets the exact scores back. They then mathematically solve for the model's secret weights.</p>
            
            <h4 style={{ marginTop: '20px', marginBottom: '8px', color: 'var(--text)' }}>2. Neural Networks</h4>
            <p>Neural networks output detailed probabilities for every class. Instead of solving equations, an attacker trains a second model using those detailed probabilities as an answer key. The second model quickly mirrors the original.</p>
            
            <h4 style={{ marginTop: '20px', marginBottom: '8px', color: 'var(--text)' }}>3. Decision Trees</h4>
            <p>Trees reveal their structure through confidence scores. Two different inputs with the exact same score probably landed on the same leaf. Attackers map out the entire tree by tweaking inputs until the score changes.</p>
            
            <h4 style={{ marginTop: '20px', marginBottom: '8px', color: 'var(--text)' }}>The Best Defense</h4>
            <p>The strongest defense is hiding the exact confidence scores. Returning only the final label forces attackers to guess blindly. That makes the attack much slower. This lab shows that extraction is still possible.</p>
          </div>
        )}
      </main>
    </div>
  );
}
