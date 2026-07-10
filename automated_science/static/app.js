let lossChart = null;
const DOMAINS = ["ecology", "oncology", "econ", "battery", "pkpd", "synbio", "climate", "cardio", "bz_chem", "epidemiology"];

function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label: 'Baseline Loss', data: [], borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', tension: 0.4, fill: true, borderWidth: 3 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: {mode: 'index', intersect: false} },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            animation: { duration: 0 }
        }
    });
}

function parseDomainName(d) {
    return d.charAt(0).toUpperCase() + d.slice(1).replace("_", " ");
}

let lastConsoleData = "";
let selectedDomain = null;

async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('current-run-id').innerText = data.current_run;
        document.getElementById('active-domain').innerText = parseDomainName(data.current_domain || 'Waiting...');
        
        // Update Console Autoscroll
        const consoleEl = document.getElementById('console-output');
        if (data.latest_log && data.latest_log !== lastConsoleData) {
            consoleEl.innerText = data.latest_log;
            consoleEl.scrollTop = consoleEl.scrollHeight;
            lastConsoleData = data.latest_log;
        }

        // Update list
        const ul = document.getElementById('domain-list');
        ul.innerHTML = '';
        DOMAINS.forEach(d => {
            const status = data.domains_status[d];
            const li = document.createElement('li');
            li.innerHTML = `<span>${parseDomainName(d)}</span> <div class="status-indicator status-${status}"></div>`;
            li.style.cursor = 'pointer';
            if (selectedDomain === d || (!selectedDomain && d === data.current_domain)) {
                li.style.border = '1px solid var(--accent)';
                li.style.background = 'rgba(56, 189, 248, 0.1)';
            }
            li.onclick = () => {
                selectedDomain = d;
                fetchStatus();
            };
            ul.appendChild(li);
        });

        // Update Chart
        let chartData = [];
        if (selectedDomain && data.histories[selectedDomain] && data.histories[selectedDomain].length > 0) {
            chartData = data.histories[selectedDomain];
        } else if (!selectedDomain && data.histories[data.current_domain] && data.histories[data.current_domain].length > 0) {
            chartData = data.histories[data.current_domain];
        } else {
            for(let d of DOMAINS) {
                if (data.histories[d]) chartData = data.histories[d];
            }
        }
        
        if (chartData.length > 0) {
            lossChart.data.labels = chartData.map(c => 'Iter ' + c.iter);
            lossChart.data.datasets[0].data = chartData.map(c => c.loss);
            lossChart.update();
        }

    } catch(e) {
        console.error("Dashboard fetch error:", e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    fetchStatus();
    setInterval(fetchStatus, 3000); // refresh every 3 seconds
});
