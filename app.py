import http.server
import json
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from core import ClientState, update_X, update_alpha
from scheduler import GradientScheduler


@dataclass
class RunConfig:
    # number of drafters is fixed to three due to GPU memory constraints
    num_clients: int = 3
    total_slots: int = 20
    cloud_capacity: int = 20
    eta: float = 0.1
    beta: float = 0.1
    initial_prompt: str = "The future of artificial intelligence depends on"
    initial_prompts: Optional[list[str]] = None
    target_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # allow construction from a dict (e.g. JSON payload)
    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        # only use known fields, ignore extras
        kwargs = {}
        for field in (
            "num_clients",
            "total_slots",
            "cloud_capacity",
            "eta",
            "beta",
            "initial_prompt",
            "initial_prompts",
            "target_model_name",
            "draft_model_name",
        ):
            if field in data:
                kwargs[field] = data[field]
        return cls(**kwargs)


class RealRunService:
    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        self._engine = None
        self._load_lock = threading.Lock()

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        with self._load_lock:
            if self._engine is None:
                from engine import RealSpeculativeEngine

                self._engine = RealSpeculativeEngine(
                    self.config.target_model_name,
                    self.config.draft_model_name,
                )
        return self._engine

    def run(self, override_config: Optional[dict] = None):
        """Execute one simulation run.

        If ``override_config`` is provided (typically from an API request),
        update the service configuration accordingly before running.
        """
        if override_config is not None:
            # merge user-supplied configuration
            cfg = RunConfig.from_dict(override_config)
            # enforce fixed drafter count
            cfg.num_clients = 3
            self.config = cfg

        engine = self._ensure_engine()
        cfg = self.config
        # Keep simulation shape stable for dashboard semantics.
        cfg.num_clients = 3
        if cfg.initial_prompts and len(cfg.initial_prompts) == cfg.num_clients:
            client_prompts = cfg.initial_prompts
        else:
            client_prompts = [cfg.initial_prompt for _ in range(cfg.num_clients)]

        scheduler = GradientScheduler(capacity=cfg.cloud_capacity)
        clients = [ClientState(eta=cfg.eta, beta=cfg.beta) for _ in range(cfg.num_clients)]

        history_allocations = np.zeros((cfg.total_slots, cfg.num_clients))
        history_goodputs = np.zeros((cfg.total_slots, cfg.num_clients))
        slot_goodput_history = []
        slot_latency_ms = []

        engine.set_prompts(client_prompts)
        initial_prompt_ids = [engine.get_prompt_ids_clone(i) for i in range(cfg.num_clients)]

        total_accepted_tokens = 0
        total_emitted_tokens = 0
        emitted_tokens_per_client = [0 for _ in range(cfg.num_clients)]
        speculative_start = time.perf_counter()

        for t in range(cfg.total_slots):
            slot_start = time.perf_counter()
            S_allocations = scheduler.allocate(clients)
            slot_total_goodput = 0
            client_step_outputs = []
            client_step_latencies = []

            for i in range(cfg.num_clients):
                S_i = int(S_allocations[i])
                client_step_start = time.perf_counter()
                l_i, new_tokens = engine.step_for_client(client_idx=i, draft_length_S=S_i)
                client_step_elapsed = time.perf_counter() - client_step_start
                engine.update_prompt_for_client(client_idx=i, new_tokens=new_tokens)
                client_step_outputs.append((S_i, l_i, len(new_tokens)))
                client_step_latencies.append(client_step_elapsed)
                total_accepted_tokens += int(l_i)
                total_emitted_tokens += int(len(new_tokens))
                emitted_tokens_per_client[i] += int(len(new_tokens))

            # Simulated parallel slot duration uses slowest client runtime.
            slot_time_s = max(client_step_latencies) if client_step_latencies else 1e-6
            slot_time_s = max(slot_time_s, 1e-6)
            for i in range(cfg.num_clients):
                S_i, l_i, _ = client_step_outputs[i]
                x_i = (1 + l_i) / slot_time_s
                tilda_alpha = l_i / S_i if S_i > 0 else 0.0
                clients[i].hat_alpha = update_alpha(clients[i].hat_alpha, tilda_alpha, clients[i].eta)
                clients[i].X = update_X(clients[i].X, x_i, clients[i].beta)

                history_allocations[t, i] = S_i
                history_goodputs[t, i] = x_i
                slot_total_goodput += x_i

            slot_goodput_history.append(float(slot_total_goodput))
            slot_latency_ms.append((time.perf_counter() - slot_start) * 1000.0)

        speculative_time_s = time.perf_counter() - speculative_start

        # Baseline: target model greedy generation for the same emitted token count.
        baseline_time_s = 0.0
        if total_emitted_tokens > 0:
            import torch

            with torch.no_grad():
                baseline_start = time.perf_counter()
                for i in range(cfg.num_clients):
                    if emitted_tokens_per_client[i] <= 0:
                        continue
                    _ = engine.target_model.generate(
                        initial_prompt_ids[i],
                        max_new_tokens=emitted_tokens_per_client[i],
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=engine.tokenizer.eos_token_id,
                    )
                baseline_time_s = time.perf_counter() - baseline_start

        speculative_toks_per_s = (
            total_emitted_tokens / speculative_time_s if speculative_time_s > 0 else 0.0
        )
        baseline_toks_per_s = total_emitted_tokens / baseline_time_s if baseline_time_s > 0 else 0.0
        speedup_ratio = (
            speculative_toks_per_s / baseline_toks_per_s if baseline_toks_per_s > 0 else 0.0
        )

        avg_allocations = np.mean(history_allocations, axis=0)
        avg_goodputs = np.mean(history_goodputs, axis=0)
        fairness = [
            float(avg_goodputs[i] / avg_allocations[i]) if avg_allocations[i] > 0 else 0.0
            for i in range(cfg.num_clients)
        ]

        final_texts = engine.decode_all()
        merged_final_text = "\n\n".join(final_texts)

        result = {
            "slot_goodput_history": slot_goodput_history,
            "slot_latency_ms": [float(v) for v in slot_latency_ms],
            "avg_allocations": avg_allocations.tolist(),
            "avg_goodputs": avg_goodputs.tolist(),
            "fairness": fairness,
            "client_labels": [f"Client {i}" for i in range(cfg.num_clients)],
            "initial_prompt": cfg.initial_prompt,
            "initial_prompts": client_prompts,
            "final_text": merged_final_text,
            "final_texts": final_texts,
            "summary": {
                "total_slots": cfg.total_slots,
                "total_emitted_tokens": total_emitted_tokens,
                "total_accepted_tokens": total_accepted_tokens,
                "speculative_time_s": speculative_time_s,
                "baseline_time_s": baseline_time_s,
                "speculative_toks_per_s": speculative_toks_per_s,
                "baseline_toks_per_s": baseline_toks_per_s,
                "speedup_ratio": speedup_ratio,
            },
        }
        # persist to local file for later inspection
        try:
            import os, datetime
            directory = os.path.join(os.getcwd(), "run_results")
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(directory, f"result_{timestamp}.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:  # keep run results even if saving fails
            pass

        return result


SERVICE = RealRunService()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Speculative Decoding Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-a: #f7f8f4;
            --bg-b: #e5ece3;
            --card: #ffffff;
            --ink: #16201b;
            --muted: #4f5d55;
            --brand: #1b8f6a;
            --brand-2: #14543f;
            --accent: #e2f3ea;
            --line: #d8e2db;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            background: radial-gradient(circle at 15% 15%, #fefef9 0%, var(--bg-a) 45%, var(--bg-b) 100%);
            min-height: 100vh;
            padding: 24px;
        }
        .wrap {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255,255,255,0.7);
            border: 1px solid var(--line);
            backdrop-filter: blur(5px);
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(23, 52, 40, 0.08);
            padding: 20px;
        }
        h1 {
            margin: 0 0 6px 0;
            font-size: 30px;
            letter-spacing: 0.2px;
        }
        .sub {
            margin: 0 0 18px 0;
            color: var(--muted);
        }
        .controls {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }
        button {
            border: 0;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--brand), var(--brand-2));
            color: white;
            padding: 10px 16px;
            font-size: 15px;
            cursor: pointer;
        }
        button:disabled {
            cursor: not-allowed;
            opacity: 0.6;
        }
        #status {
            color: var(--muted);
            font-size: 14px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 10px;
            margin: 12px 0 20px;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px 12px;
        }
        .label {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 2px;
        }
        .value {
            font-size: 21px;
            font-weight: 700;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .panel {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 10px;
            height: 320px;
            overflow: hidden;
        }
        .panel h3 {
            margin: 4px 0 12px;
            font-size: 15px;
        }
        .panel canvas {
            width: 100% !important;
            height: 260px !important;
            display: block;
        }
        .text-panel {
            margin-top: 14px;
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 12px;
        }
        #finalText {
            margin-top: 6px;
            white-space: pre-wrap;
            line-height: 1.5;
            color: #26352c;
        }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Real Speculative Decoding Dashboard</h1>
        <p class="sub">Runs real model inference (Qwen 7B target + 0.5B draft), records runtime stats, and compares against target-only baseline.</p>

        <div class="controls">
            <!-- configuration inputs -->
            <label>Clients: <input id="numClients" type="number" value="3" min="1" style="width:60px" disabled></label> <!-- fixed at 3 -->
            <label>Slots: <input id="totalSlots" type="number" value="20" min="1" style="width:60px"></label>
            <label>Capacity: <input id="cloudCapacity" type="number" value="20" min="1" style="width:60px"></label>
            <label>η: <input id="eta" type="number" step="0.01" value="0.1" style="width:50px"></label>
            <label>β: <input id="beta" type="number" step="0.01" value="0.1" style="width:50px"></label>
            <label>Prompt: <input id="initialPrompt" type="text" value="The future of artificial intelligence depends on" style="width:300px"></label>
            <button id="runBtn">Run Real Experiment</button>
            <a id="saveBtn" href="#" style="display:none; margin-left:12px;">Download Results</a>
            <span id="status">Idle</span>
        </div>

        <div class="stats" id="stats"></div>

        <div class="grid">
            <div class="panel">
                <h3>Total Goodput Per Slot</h3>
                <canvas id="goodputChart"></canvas>
            </div>
            <div class="panel">
                <h3>Slot Latency (ms)</h3>
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="panel">
                <h3>Average Allocation (S_i)</h3>
                <canvas id="allocChart"></canvas>
            </div>
            <div class="panel">
                <h3>Fairness (x/S)</h3>
                <canvas id="fairnessChart"></canvas>
            </div>
        </div>

        <div class="text-panel">
            <div class="label">Final Generated Text</div>
            <div id="finalText">No run yet.</div>
        </div>
    </div>

    <script>
        const goodputCtx = document.getElementById('goodputChart').getContext('2d');
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        const allocCtx = document.getElementById('allocChart').getContext('2d');
        const fairnessCtx = document.getElementById('fairnessChart').getContext('2d');
        const runBtn = document.getElementById('runBtn');
        const saveBtn = document.getElementById('saveBtn');
        const statusEl = document.getElementById('status');
        const statsEl = document.getElementById('stats');
        const finalTextEl = document.getElementById('finalText');
        // config inputs
        const numClientsInp = document.getElementById('numClients');
        const totalSlotsInp = document.getElementById('totalSlots');
        const cloudCapacityInp = document.getElementById('cloudCapacity');
        const etaInp = document.getElementById('eta');
        const betaInp = document.getElementById('beta');
        const initialPromptInp = document.getElementById('initialPrompt');

        let goodputChart;
        let latencyChart;
        let allocChart;
        let fairnessChart;

        function makeLineChart(ctx, label, color) {
            return new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [{ label, data: [], borderColor: color, tension: 0.2, fill: false }] },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        function makeBarChart(ctx, label, color) {
            return new Chart(ctx, {
                type: 'bar',
                data: { labels: [], datasets: [{ label, data: [], backgroundColor: color }] },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        function initCharts() {
            if (goodputChart) goodputChart.destroy();
            if (latencyChart) latencyChart.destroy();
            if (allocChart) allocChart.destroy();
            if (fairnessChart) fairnessChart.destroy();

            goodputChart = makeLineChart(goodputCtx, 'Goodput', '#1b8f6a');
            latencyChart = makeLineChart(latencyCtx, 'Latency (ms)', '#9d6128');
            allocChart = makeBarChart(allocCtx, 'Avg S_i', 'rgba(20,84,63,0.75)');
            fairnessChart = makeBarChart(fairnessCtx, 'x/S', 'rgba(40,123,163,0.75)');
        }

        function statCard(label, value) {
            return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
        }

        function escapeHtml(text) {
            return String(text)
                .replaceAll('&', '&amp;')
                .replaceAll('<', '&lt;')
                .replaceAll('>', '&gt;')
                .replaceAll('"', '&quot;')
                .replaceAll("'", '&#39;');
        }

        function renderFinalTexts(data) {
            if (Array.isArray(data.final_texts) && data.final_texts.length > 0) {
                const blocks = data.final_texts.map((text, idx) => {
                    return `<div style="margin-bottom:16px;">
                        <div class="label">Client ${idx}</div>
                        <div style="white-space:pre-wrap; line-height:1.5;">${escapeHtml(text)}</div>
                    </div>`;
                });
                finalTextEl.innerHTML = blocks.join('');
                return;
            }
            finalTextEl.textContent = data.final_text || '(empty)';
        }

        function setStats(summary) {
            const speed = summary.speedup_ratio > 0 ? `${summary.speedup_ratio.toFixed(2)}x` : 'N/A';
            statsEl.innerHTML = [
                statCard('Total Slots', summary.total_slots),
                statCard('Emitted Tokens', summary.total_emitted_tokens),
                statCard('Accepted Tokens', summary.total_accepted_tokens),
                statCard('Speculative Time', `${summary.speculative_time_s.toFixed(2)}s`),
                statCard('Baseline Time', `${summary.baseline_time_s.toFixed(2)}s`),
                statCard('Speculative tok/s', summary.speculative_toks_per_s.toFixed(2)),
                statCard('Baseline tok/s', summary.baseline_toks_per_s.toFixed(2)),
                statCard('Speedup', speed),
            ].join('');
        }

        function updateCharts(data) {
            const labels = Array.from({ length: data.slot_goodput_history.length }, (_, i) => i + 1);

            goodputChart.data.labels = labels;
            goodputChart.data.datasets[0].data = data.slot_goodput_history;
            goodputChart.update();

            latencyChart.data.labels = labels;
            latencyChart.data.datasets[0].data = data.slot_latency_ms;
            latencyChart.update();

            allocChart.data.labels = data.client_labels;
            allocChart.data.datasets[0].data = data.avg_allocations;
            allocChart.update();

            fairnessChart.data.labels = data.client_labels;
            fairnessChart.data.datasets[0].data = data.fairness;
            fairnessChart.update();
        }

        initCharts();

        runBtn.addEventListener('click', async () => {
            runBtn.disabled = true;
            statusEl.textContent = 'Running real model inference... this can take a while.';
            saveBtn.style.display = 'none';
            try {
                const config = {
                    num_clients: parseInt(numClientsInp.value, 10),
                    total_slots: parseInt(totalSlotsInp.value, 10),
                    cloud_capacity: parseInt(cloudCapacityInp.value, 10),
                    eta: parseFloat(etaInp.value),
                    beta: parseFloat(betaInp.value),
                    initial_prompt: initialPromptInp.value,
                };
                const response = await fetch('/api/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({config}),
                });
                const data = await response.json();
                setStats(data.summary);
                updateCharts(data);
                renderFinalTexts(data);
                statusEl.textContent = 'Done';

                // prepare download link
                const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                saveBtn.href = url;
                saveBtn.download = `run_${Date.now()}.json`;
                saveBtn.style.display = 'inline-block';
            } catch (err) {
                statusEl.textContent = `Failed: ${err.message}`;
            } finally {
                runBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # keep GET route for backwards compatibility, but warn that POST is preferred
        if self.path == "/api/run":
            try:
                sim_data = SERVICE.run()
                body = json.dumps(sim_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            return

        self.send_error(404, "File Not Found")

    def do_POST(self):
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(post_data) if post_data else {}
            except json.JSONDecodeError:
                payload = {}

            try:
                sim_data = SERVICE.run(payload.get("config"))
                body = json.dumps(sim_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            return

        self.send_error(404, "File Not Found")


def run_server(port=8000):
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
