"""Q-Portfolio dark observatory dashboard."""
import html
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import ROOT


st.set_page_config(page_title="Q-Portfolio | Command Center", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --void:#000; --card:#050607; --soft:#0b0c0e; --white:#fff; --bone:#e8eaee; --ash:#a6a8ad; --steel:#64666d; --line:#202126; --violet:#9984d8; --mint:#3fcb7f; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; }
.stApp { background:var(--void); color:var(--bone); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stToolbar"] { visibility:visible; }
[data-testid="stSidebarCollapsedControl"] { display:block!important; visibility:visible!important; }
[data-testid="stSidebarCollapsedControl"] button { background:var(--card); border:1px solid var(--line); color:var(--white); }
.block-container { max-width:1240px; padding:28px 5vw 64px; }
[data-testid="stSidebar"] { background:#030405; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:var(--ash); }
h1, h2, h3, p { letter-spacing:-.03em; }
h1 { font-size:clamp(2rem,4vw,3rem)!important; line-height:1!important; font-weight:500!important; color:var(--white)!important; }
h2 { font-size:1.65rem!important; line-height:1.1!important; font-weight:500!important; color:var(--white)!important; }
.eyebrow { color:var(--violet); font:500 .72rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
.topline { border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:34px; }
.brand { color:var(--white); font-size:1rem; font-weight:600; letter-spacing:-.03em; }
.brand-mark { color:var(--mint); margin-right:8px; }
.status { color:var(--mint); font:500 .72rem 'DM Mono',monospace; text-align:right; padding-top:4px; }
.status-dot { display:inline-block; width:7px; height:7px; background:var(--mint); border-radius:50%; margin-right:7px; box-shadow:0 0 12px rgba(63,203,127,.65); }
.hero-copy { padding:18px 0 28px; }
.hero-copy p { max-width:620px; color:var(--ash); font-size:1rem; line-height:1.55; }
.legend { border-left:1px solid var(--violet); padding:4px 0 4px 18px; margin-top:16px; }
.legend-title { color:var(--white); font-size:.82rem; font-weight:600; margin-bottom:8px; }
.legend-line { color:var(--ash); font-size:.76rem; line-height:1.55; margin:3px 0; }
.legend-line b { color:var(--bone); font-family:'DM Mono',monospace; font-weight:500; }
.section-rule { border-top:1px solid var(--line); margin:34px 0 22px; padding-top:22px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:18px 20px; min-height:112px; }
.card-label { color:var(--ash); font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
.card-value { color:var(--white); font:500 1.7rem 'DM Mono',monospace; margin-top:13px; }
.card-detail { color:var(--mint); font:400 .72rem 'DM Mono',monospace; margin-top:7px; }
.panel { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:4px; }
.panel-head { padding:14px 16px 4px; }
.panel-kicker { color:var(--steel); font:400 .68rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.1em; }
.panel-title { color:var(--white); font-size:1.05rem; margin-top:5px; }
.stDataFrame { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
div[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:17px 18px; }
div[data-testid="stMetricLabel"] { color:var(--ash); }
div[data-testid="stMetricValue"] { color:var(--white); font-family:'DM Mono',monospace; }
.stButton > button { background:var(--white); color:var(--void); border:0; border-radius:6px; font-weight:600; min-height:42px; }
.stButton > button:hover { background:var(--bone); color:var(--void); }
label { color:var(--ash)!important; font-size:.78rem!important; }
.sidebar-title { color:var(--white); font-size:1.1rem; font-weight:600; margin:8px 0 28px; }
.sidebar-note { border-top:1px solid var(--line); margin-top:26px; padding-top:18px; color:var(--steel); font-size:.73rem; line-height:1.6; }
@media (max-width:700px) { .block-container { padding:18px 20px 48px; } .status { text-align:left; margin-top:12px; } }
</style>
""",
    unsafe_allow_html=True,
)

results_dir = ROOT / "results"


def network_visual(stage="loading", ticker="", epoch=0, total=0):
    active_layer = {"loading": 0, "features": 1, "classical": 2, "quantum": 2, "scoring": 3, "allocation": 3, "backtest": 3, "complete": 3}.get(stage, 1)
    columns = [(34, "INPUT"), (154, "FEATURE MAP"), (274, "VARIATIONAL"), (394, "OUTPUT")]
    rows = [54, 108, 162, 216]
    lines = []
    for left, _ in columns[:-1]:
        for y_start in rows:
            for y_end in rows:
                lines.append(f'<line class="connection" x1="{left + 14}" y1="{y_start}" x2="{left + 106}" y2="{y_end}" />')
    nodes = []
    for layer_index, (x, label) in enumerate(columns):
        for row_index, y in enumerate(rows):
            state = "active" if layer_index == active_layer else ""
            nodes.append(f'<circle class="node {state}" cx="{x}" cy="{y}" r="9" /><text class="node-label" x="{x + 18}" y="{y + 4}" text-anchor="start">{row_index + 1}</text>')
    progress_text = f"epoch {epoch}/{total}" if total else "signal path active"
    return f'''<div class="network-shell">
<style>
.network-shell {{ background:#050607; border:1px solid #202126; border-radius:16px; padding:18px 18px 12px; margin:12px 0 22px; }}
.network-head {{ display:flex; justify-content:space-between; color:#a6a8ad; font:500 .7rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }}
.network-head strong {{ color:#3fcb7f; font-weight:500; }}
.network-svg {{ width:100%; min-height:220px; margin-top:10px; overflow:visible; }}
.connection {{ stroke:#514a76; stroke-width:1; opacity:.38; stroke-dasharray:5 7; animation:flow 1.8s linear infinite; }}
.node {{ fill:#0b0c0e; stroke:#64666d; stroke-width:1.5; transition:all .3s ease; }}
.node.active {{ fill:#9984d8; stroke:#fff; stroke-width:2; animation:pulse 1.2s ease-in-out infinite; }}
.node-label {{ fill:#64666d; font:400 9px 'DM Mono',monospace; }}
.layer-label {{ fill:#a6a8ad; font:500 8px 'DM Mono',monospace; letter-spacing:1px; }}
@keyframes flow {{ to {{ stroke-dashoffset:-24; }} }}
@keyframes pulse {{ 0%,100% {{ r:9; opacity:.75; }} 50% {{ r:12; opacity:1; }} }}
</style>
<div class="network-head"><span>{html.escape(ticker or 'PORTFOLIO SCAN')}</span><strong>{html.escape(progress_text)}</strong></div>
<svg class="network-svg" viewBox="0 0 428 252" role="img" aria-label="Animated model signal path">{''.join(lines)}{''.join(nodes)}{''.join(f'<text class="layer-label" x="{x}" y="246" text-anchor="middle">{label}</text>' for x, label in columns)}</svg>
</div>'''


def stage_list(current_stage):
    stages = [("loading", "Load OHLCV data"), ("features", "Calculate technical features"), ("classical", "Fit classical baseline"), ("quantum", "Train variational circuit"), ("scoring", "Score held-out predictions"), ("allocation", "Optimize portfolio weights"), ("backtest", "Run walk-forward backtest")]
    rows = []
    order = [stage[0] for stage in stages]
    current_index = order.index(current_stage) if current_stage in order else -1
    for index, (key, label) in enumerate(stages):
        marker = "DONE" if index < current_index or current_stage == "complete" else "RUN" if key == current_stage else "WAIT"
        rows.append(f'<div class="stage-row"><span class="stage-marker {marker.lower()}">{marker}</span><span>{label}</span></div>')
    return '<div class="stage-list">' + ''.join(rows) + '</div>'


with st.sidebar:
    st.markdown('<div class="sidebar-title"><span class="brand-mark">◈</span>Q-PORTFOLIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Control surface</div>', unsafe_allow_html=True)
    mode = st.selectbox("Data source", ["sample", "market"])
    epochs = st.slider("Quantum epochs", 1, 30, 20)
    run_analysis = st.button("Run analysis", use_container_width=True)
    st.markdown('<div class="sidebar-note">Signals are evaluated chronologically. Outputs are written to the results directory after each run.<br><br>Not investment advice.</div>', unsafe_allow_html=True)

if run_analysis:
    st.markdown('<div class="section-rule"><div class="eyebrow">Live execution trace</div><h2>Scanning the universe</h2></div>', unsafe_allow_html=True)
    progress_placeholder = st.empty()
    network_placeholder = st.empty()
    stage_placeholder = st.empty()
    progress_placeholder.progress(0, text="Preparing scan engine...")
    network_placeholder.markdown(network_visual(), unsafe_allow_html=True)
    process = subprocess.Popen([sys.executable, str(ROOT / "run_pipeline.py"), "--mode", mode, "--epochs", str(epochs)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=ROOT)
    output = []
    for raw_line in process.stdout:
        line = raw_line.strip()
        if line.startswith("QPROGRESS|"):
            _, stage, detail, percent, ticker, epoch, total = line.split("|", 6)
            progress_placeholder.progress(float(percent), text=detail)
            network_placeholder.markdown(network_visual(stage, ticker, int(epoch), int(total)), unsafe_allow_html=True)
            stage_placeholder.markdown(stage_list(stage), unsafe_allow_html=True)
        elif line:
            output.append(line)
    return_code = process.wait()
    if return_code:
        st.error("Scan failed\n\n" + "\n".join(output[-40:]))
    else:
        progress_placeholder.progress(1.0, text="Analysis complete")
        network_placeholder.markdown(network_visual("complete", "ALL ASSETS"), unsafe_allow_html=True)
        stage_placeholder.markdown(stage_list("complete"), unsafe_allow_html=True)
        st.success("Analysis complete")

model_path = results_dir / "model_comparison.csv"
allocation_path = results_dir / "portfolio_allocation.csv"
backtest_path = results_dir / "backtest_results.csv"
summary_path = results_dir / "backtest_summary.csv"
comparison_path = results_dir / "performance_comparison.csv"
model_data = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
allocation_data = pd.read_csv(allocation_path) if allocation_path.exists() else pd.DataFrame()
summary_data = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
comparison_data = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()

brand_col, status_col = st.columns([3, 1])
with brand_col:
    st.markdown('<div class="topline"><div class="brand"><span class="brand-mark">◈</span>Q-PORTFOLIO <span style="color:#64666d;font-weight:400">/ COMMAND CENTER</span></div></div>', unsafe_allow_html=True)
with status_col:
    st.markdown('<div class="topline"><div class="status"><span class="status-dot"></span>LOCAL ENGINE READY</div></div>', unsafe_allow_html=True)

hero_col, legend_col = st.columns([3, 2])
with hero_col:
    st.markdown('<div class="hero-copy"><div class="eyebrow">Hybrid intelligence / 04 qubits</div>', unsafe_allow_html=True)
    st.title("Signals for the next move.")
    st.markdown('<p>Compare classical and quantum classifiers, inspect their portfolio implications, and trace the walk-forward result from one quiet command surface.</p></div>', unsafe_allow_html=True)
with legend_col:
    st.markdown('<div class="legend"><div class="legend-title">METRIC LEGEND</div><div class="legend-line"><b>F1</b> harmonic mean of precision and recall</div><div class="legend-line"><b>CONFUSION MATRIX</b> true / false prediction counts</div><div class="legend-line"><b>ROC AUC</b> ranking quality across thresholds</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"><div class="eyebrow">At a glance</div><h2>Market posture</h2></div>', unsafe_allow_html=True)
metric_cols = st.columns(4)
if not allocation_data.empty:
    values = [("Capital deployed", f"{allocation_data['portfolio_weight'].sum():.0%}", "fully allocated"), ("Expected return", f"{allocation_data['expected_return'].mean():.2%}", "per rebalance"), ("Largest exposure", allocation_data.loc[allocation_data['portfolio_weight'].idxmax(), 'asset'], f"{allocation_data['portfolio_weight'].max():.0%} weight"), ("Universe", str(len(allocation_data)), "tracked assets")]
else:
    values = [("Capital deployed", "--", "run an analysis"), ("Expected return", "--", "awaiting signal"), ("Largest exposure", "--", "awaiting signal"), ("Universe", "--", "awaiting data")]
for column, (label, value, detail) in zip(metric_cols, values):
    with column:
        st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-detail">{detail}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"><div class="eyebrow">Model observatory</div><h2>Prediction quality</h2></div>', unsafe_allow_html=True)
if not model_data.empty:
    chart_col, table_col = st.columns([1.05, 1.4])
    with chart_col:
        chart_data = model_data.groupby("model", as_index=False)[["accuracy", "f1", "roc_auc"]].mean().melt(id_vars="model", var_name="metric", value_name="score")
        chart = px.bar(chart_data, x="metric", y="score", color="model", barmode="group", range_y=[0, 1], color_discrete_map={"classical": "#9984d8", "quantum": "#3fcb7f"})
        chart.update_layout(template="plotly_dark", paper_bgcolor="#050607", plot_bgcolor="#050607", margin=dict(l=12, r=12, t=22, b=12), height=300, font=dict(family="Space Grotesk", color="#a6a8ad"), legend=dict(orientation="h", y=1.12, x=0), xaxis_title=None, yaxis_title=None)
        chart.update_yaxes(gridcolor="#202126", tickformat=".0%")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    with table_col:
        st.markdown('<div class="panel"><div class="panel-head"><div class="panel-kicker">Per-asset diagnostics</div><div class="panel-title">Classifier comparison</div></div></div>', unsafe_allow_html=True)
        display_model = model_data.drop(columns=["confusion_matrix"], errors="ignore").copy()
        for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if column in display_model:
                display_model[column] = display_model[column].round(3)
        st.dataframe(display_model, width="stretch", hide_index=True, height=300)
else:
    st.info("Run an analysis from the control surface to populate model diagnostics.")

st.markdown('<div class="section-rule"><div class="eyebrow">Allocation desk</div><h2>Portfolio construction</h2></div>', unsafe_allow_html=True)
if not allocation_data.empty:
    allocation_col, table_col = st.columns([1, 1.25])
    with allocation_col:
        pie = px.pie(allocation_data, names="asset", values="portfolio_weight", hole=.68, color_discrete_sequence=["#9984d8", "#3fcb7f", "#e8eaee", "#64666d", "#514a76"])
        pie.update_layout(template="plotly_dark", paper_bgcolor="#050607", plot_bgcolor="#050607", margin=dict(l=8, r=8, t=12, b=12), height=330, font=dict(family="Space Grotesk", color="#a6a8ad"), legend=dict(orientation="h", y=-.05, x=0))
        st.plotly_chart(pie, width="stretch", config={"displayModeBar": False})
    with table_col:
        display_allocation = allocation_data.copy()
        for column in ["prediction_probability", "expected_return", "portfolio_weight"]:
            if column in display_allocation:
                display_allocation[column] = display_allocation[column].round(3)
        st.dataframe(display_allocation, width="stretch", hide_index=True, height=330)
else:
    st.info("Portfolio allocation will appear here after the pipeline completes.")

st.markdown('<div class="section-rule"><div class="eyebrow">Walk-forward record</div><h2>Performance trace</h2></div>', unsafe_allow_html=True)
if backtest_path.exists():
    backtest_data = pd.read_csv(backtest_path)
    trace = px.line(backtest_data, x="date", y="portfolio_value", color="strategy", color_discrete_sequence=["#9984d8", "#3fcb7f", "#e8eaee"])
    trace.update_layout(template="plotly_dark", paper_bgcolor="#050607", plot_bgcolor="#050607", margin=dict(l=12, r=12, t=18, b=12), height=360, font=dict(family="Space Grotesk", color="#a6a8ad"), legend=dict(orientation="h", y=1.08, x=0), xaxis_title=None, yaxis_title=None)
    trace.update_xaxes(gridcolor="#202126")
    trace.update_yaxes(gridcolor="#202126")
    st.plotly_chart(trace, width="stretch", config={"displayModeBar": False})
if not summary_data.empty:
    summary_display = summary_data.drop(columns=["daily_returns"], errors="ignore").copy()
    for column in ["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "cumulative_return"]:
        if column in summary_display:
            summary_display[column] = summary_display[column].round(3)
    st.dataframe(summary_display, width="stretch", hide_index=True)

if not comparison_data.empty:
    st.markdown('<div class="section-rule"><div class="eyebrow">Classical / quantum delta</div><h2>Risk and runtime comparison</h2></div>', unsafe_allow_html=True)
    display_comparison = comparison_data.copy()
    for column in ["expected_return", "volatility", "sharpe_ratio", "max_drawdown", "runtime_seconds"]:
        if column in display_comparison:
            display_comparison[column] = display_comparison[column].round(4)
    st.dataframe(display_comparison, width="stretch", hide_index=True)

