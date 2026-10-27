/**
 * URsafe Frontend Application Logic
 * Integrates Leaflet Maps, Chart.js, REST APIs, AI Companion Guidance, and Multi-Route Comparison.
 */

// State Management
const state = {
  predictMap: null,
  predictMarker: null,
  routeMap: null,
  routeLayers: [],
  selectedRouteId: 'fast',
  predictProfile: 'hiker',
  routeProfile: 'hiker',
  rainfallChart: null,
  featuresChart: null,
  clustersChart: null,
  currentCoords: { lat: 30.3244, lon: 78.0339 } // Default: Dehradun, Uttarakhand
};

// Band Color Mappings
const BAND_COLORS = {
  Green: '#10b981',
  Yellow: '#f59e0b',
  Orange: '#f97316',
  Red: '#ef4444'
};

// Create SVG Pin Marker
function createPinIcon(bandName) {
  const color = BAND_COLORS[bandName] || '#0f172a';
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 36" width="28" height="42">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 9 12 24 12 24s12-15 12-24c0-6.63-5.37-12-12-12z" fill="${color}" stroke="#ffffff" stroke-width="2"/>
      <circle cx="12" cy="12" r="5" fill="#ffffff"/>
    </svg>
  `;
  return L.divIcon({
    className: 'custom-pin-marker',
    html: svg,
    iconSize: [28, 42],
    iconAnchor: [14, 42],
    popupAnchor: [0, -38]
  });
}

// -----------------------------------------------------------------------------
// 1. INITIALIZATION
// -----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initPredictMap();
  initRouteMap();
  bindUIEvents();
  loadReplayEvents();
  loadModelReport();

  // Initial prediction at default coordinate
  fetchPrediction(state.currentCoords.lat, state.currentCoords.lon);
});

// -----------------------------------------------------------------------------
// 2. PREDICT SECTION LOGIC
// -----------------------------------------------------------------------------
function initPredictMap() {
  state.predictMap = L.map('predict-map', {
    center: [state.currentCoords.lat, state.currentCoords.lon],
    zoom: 7,
    zoomControl: true
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(state.predictMap);

  // Map Click Listener
  state.predictMap.on('click', (e) => {
    const { lat, lng } = e.latlng;
    fetchPrediction(lat, lng);
  });
}

async function fetchPrediction(lat, lon, dateOverride = null) {
  const dateInput = document.getElementById('predict-date-input');
  const dateVal = dateOverride || (dateInput.value ? dateInput.value : null);

  state.currentCoords = { lat, lon };

  // Update map marker
  if (state.predictMarker) {
    state.predictMap.removeLayer(state.predictMarker);
  }
  state.predictMarker = L.marker([lat, lon], {
    icon: createPinIcon('Green')
  }).addTo(state.predictMap);

  try {
    const payload = {
      lat: lat,
      lon: lon,
      profile: state.predictProfile,
      date: dateVal
    };

    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error('Prediction API failure');
    const data = await resp.json();
    renderPredictionResults(data);

    // Update pin color by band
    state.predictMarker.setIcon(createPinIcon(data.band));
    state.predictMarker.bindPopup(`<b>${data.band_label}</b> (${data.score}/100)<br>${lat.toFixed(4)}, ${lon.toFixed(4)}`).openPopup();
  } catch (err) {
    console.error('Error fetching prediction:', err);
  }
}

function renderPredictionResults(data) {
  document.getElementById('predict-empty-state').style.display = 'none';
  const card = document.getElementById('predict-card');
  card.style.display = 'block';

  // Score & Band
  document.getElementById('res-score').innerText = data.score;
  document.getElementById('res-confidence').innerText = data.confidence;

  const bandBadge = document.getElementById('res-band-badge');
  bandBadge.className = `band-pill-badge band-${data.band}`;
  bandBadge.innerText = `${data.band} (${data.band_label})`;

  // Companion Advice
  document.getElementById('res-companion-text').innerText = data.advice;

  // Terrain Zone
  document.getElementById('res-zone-name').innerText = data.terrain_zone.name;
  const membershipsDiv = document.getElementById('res-zone-memberships');
  membershipsDiv.innerHTML = '';
  if (data.terrain_zone.memberships) {
    for (const [zName, prob] of Object.entries(data.terrain_zone.memberships)) {
      const span = document.createElement('span');
      span.innerText = `${zName}: ${(prob * 100).toFixed(0)}%`;
      membershipsDiv.appendChild(span);
    }
  }

  // Top SHAP Factors List
  const factorsList = document.getElementById('res-factors-list');
  factorsList.innerHTML = '';
  (data.top_factors || []).forEach(f => {
    const li = document.createElement('li');
    li.className = 'factor-item';
    const arrow = f.direction === 'up' ? '&#8593;' : '&#8595;';
    const arrowColor = f.direction === 'up' ? '#ef4444' : '#10b981';

    li.innerHTML = `
      <div class="factor-name-desc">
        <div><strong>${f.label}</strong> <span class="tag-badge tag-${f.provenance}">${f.provenance}</span></div>
        <div style="color: var(--color-muted-text); font-size: 12px; margin-top: 2px;">${f.description}</div>
      </div>
      <div style="color: ${arrowColor}; font-weight: 700; font-size: 16px;">${arrow}</div>
    `;
    factorsList.appendChild(li);
  });

  // Freshness
  document.getElementById('res-freshness').innerText = `${data.date} &bull; Model ${data.model_version}`;

  // Rainfall Chart
  renderRainfallChart(data.rainfall_chart);
}

function renderRainfallChart(chartData) {
  const ctx = document.getElementById('predict-rainfall-chart').getContext('2d');
  if (state.rainfallChart) {
    state.rainfallChart.destroy();
  }

  if (!chartData || chartData.length === 0) return;

  const labels = chartData.map(d => d.date.slice(5)); // MM-DD
  const values = chartData.map(d => d.rainfall_mm);
  const colors = chartData.map(d => d.type === 'forecast' ? '#f59e0b' : '#0284c7');

  state.rainfallChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Daily Precipitation (mm)',
        data: values,
        backgroundColor: colors,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => chartData[items[0].dataIndex].date,
            label: (item) => `${item.raw} mm (${chartData[item.dataIndex].type})`
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: 8, font: { size: 10 } }
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// 3. ROUTE COMPARISON LOGIC
// -----------------------------------------------------------------------------
function initRouteMap() {
  state.routeMap = L.map('route-map', {
    center: [30.1, 78.5],
    zoom: 8,
    zoomControl: true
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(state.routeMap);
}

async function executeRouteComparison() {
  const btn = document.getElementById('btn-compare-routes');
  const loading = document.getElementById('route-loading');
  const cardsList = document.getElementById('route-cards-list');

  btn.disabled = true;
  loading.style.display = 'block';
  cardsList.innerHTML = '';

  // Clear existing layers
  state.routeLayers.forEach(l => state.routeMap.removeLayer(l));
  state.routeLayers = [];

  try {
    // Default cities: Rishikesh to Joshimath in the Himalayas
    const startCoord = [30.0869, 78.2676];
    const endCoord = [30.5574, 79.5667];

    const resp = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start: startCoord,
        end: endCoord,
        profile: state.routeProfile
      })
    });

    const data = await resp.json();
    loading.style.display = 'none';
    btn.disabled = false;

    if (data.routes && data.routes.length > 0) {
      renderRouteCards(data.routes);
      drawRouteOnMap(data.routes[0]); // Draw first route by default
    }
  } catch (err) {
    loading.style.display = 'none';
    btn.disabled = false;
    console.error('Error calculating routes:', err);
  }
}

function renderRouteCards(routes) {
  const container = document.getElementById('route-cards-list');
  container.innerHTML = '';

  routes.forEach((r, idx) => {
    const card = document.createElement('div');
    card.className = `route-card ${idx === 0 ? 'selected' : ''}`;
    card.id = `route-card-${r.id}`;

    card.innerHTML = `
      <div class="route-card-header">
        <div>
          <span class="route-archetype">${r.archetype}</span>
          <span style="font-size: 11px; padding: 2px 8px; border-radius: 9999px; background: #f1f5f9; margin-left: 8px;">${r.badge}</span>
        </div>
        <div style="font-weight: 600; font-size: 15px;">${r.duration_min} min</div>
      </div>
      <div style="font-size: 13px; color: var(--color-muted-text);">${r.why}</div>
      <div class="route-stats-grid">
        <div>
          <span style="color: var(--color-muted-text);">Distance</span><br>
          <span class="route-stat-val">${r.distance_km} km</span>
        </div>
        <div>
          <span style="color: var(--color-muted-text);">Mean Hazard</span><br>
          <span class="route-stat-val">${r.mean_risk}/100</span>
        </div>
        <div>
          <span style="color: var(--color-muted-text);">Peak Hazard</span><br>
          <span class="route-stat-val" style="color: ${BAND_COLORS[r.max_risk > 75 ? 'Red' : (r.max_risk > 50 ? 'Orange' : 'Green')]};">${r.max_risk}/100</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      document.querySelectorAll('.route-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      drawRouteOnMap(r);
    });

    container.appendChild(card);
  });
}

function drawRouteOnMap(routeObj) {
  // Clear previous layers
  state.routeLayers.forEach(l => state.routeMap.removeLayer(l));
  state.routeLayers = [];

  const bounds = L.latLngBounds();

  (routeObj.segments || []).forEach(seg => {
    const color = BAND_COLORS[seg.properties.band] || '#3b82f6';
    const coords = seg.geometry.coordinates.map(c => [c[1], c[0]]); // GeoJSON [lon, lat] -> Leaflet [lat, lon]

    const polyline = L.polyline(coords, {
      color: color,
      weight: 6,
      opacity: 0.85
    }).addTo(state.routeMap);

    polyline.bindPopup(`
      <strong>Segment Hazard: ${seg.properties.score}/100 (${seg.properties.band})</strong><br>
      Slope: ${seg.properties.slope_deg}&deg; &bull; 3-Day Rain: ${seg.properties.rain_3d} mm
    `);

    coords.forEach(c => bounds.extend(c));
    state.routeLayers.push(polyline);
  });

  if (bounds.isValid()) {
    state.routeMap.fitBounds(bounds, { padding: [40, 40] });
  }
}

// -----------------------------------------------------------------------------
// 4. MODEL REPORT CARD LOGIC
// -----------------------------------------------------------------------------
async function loadModelReport() {
  try {
    const [reportResp, datasetsResp] = await Promise.all([
      fetch('/api/model/report'),
      fetch('/api/datasets')
    ]);

    const report = await reportResp.json();
    const datasets = await datasetsResp.json();

    // Populate Dataset Stats
    if (datasets.sources && datasets.sources.length > 0) {
      const glc = datasets.sources.find(s => s.dataset_name.includes('Landslide'));
      if (glc) {
        document.getElementById('meta-positives').innerText = `${glc.rows.toLocaleString()} events`;
        document.getElementById('meta-negatives').innerText = `${(glc.rows * 2).toLocaleString()} pseudo-absences`;
      }
    }

    // Populate Models Comparison Table
    const tbody = document.getElementById('models-benchmark-body');
    tbody.innerHTML = '';
    const comp = report.performance?.models_comparison || {};
    for (const [mName, mStats] of Object.entries(comp)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${mName}</strong></td>
        <td>${mStats.roc_auc?.toFixed(3) || '--'}</td>
        <td>${mStats.pr_auc?.toFixed(3) || '--'}</td>
        <td>${mStats.f1?.toFixed(3) || '--'}</td>
        <td>${mStats.recall?.toFixed(3) || '--'}</td>
      `;
      tbody.appendChild(tr);
    }

    // Feature Importances Chart
    if (report.feature_importances) {
      renderFeatureImportances(report.feature_importances);
    }

    // Clusters PCA Chart
    if (report.clusters && report.clusters.sample_points) {
      renderClustersPCA(report.clusters);
    }
  } catch (err) {
    console.error('Error loading model report:', err);
  }
}

function renderFeatureImportances(featDict) {
  const ctx = document.getElementById('model-features-chart').getContext('2d');
  const sorted = Object.entries(featDict).sort((a, b) => b[1] - a[1]).slice(0, 8);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(s => s[0].replace('_', ' ')),
      datasets: [{
        label: 'Relative Predictive Importance',
        data: sorted.map(s => s[1]),
        backgroundColor: '#0f172a',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } },
        y: { ticks: { font: { size: 11 } } }
      }
    }
  });
}

function renderClustersPCA(clustersData) {
  const ctx = document.getElementById('model-clusters-chart').getContext('2d');
  const points = clustersData.sample_points || [];

  const datasets = [
    {
      label: 'Terrain Points',
      data: points.map(p => ({ x: p.x, y: p.y })),
      backgroundColor: points.map(p => p.is_landslide ? '#ef4444' : '#10b981'),
      pointRadius: 4,
      pointHoverRadius: 6
    },
    {
      label: 'Centroids',
      data: (clustersData.centroids || []).map(c => ({ x: c.x, y: c.y })),
      backgroundColor: '#000000',
      pointStyle: 'triangle',
      pointRadius: 8
    }
  ];

  new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 10 } } }
      },
      scales: {
        x: { title: { display: true, text: 'PCA Component 1' }, grid: { color: 'rgba(0,0,0,0.05)' } },
        y: { title: { display: true, text: 'PCA Component 2' }, grid: { color: 'rgba(0,0,0,0.05)' } }
      }
    }
  });
}

// -----------------------------------------------------------------------------
// 5. EVENT LISTENERS & SEARCH
// -----------------------------------------------------------------------------
function bindUIEvents() {
  // Traveler Profile Chips for Predict
  document.querySelectorAll('#predict-profile-chips .chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#predict-profile-chips .chip-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.predictProfile = btn.dataset.profile;
      fetchPrediction(state.currentCoords.lat, state.currentCoords.lon);
    });
  });

  // Traveler Profile Chips for Routing
  document.querySelectorAll('#route-profile-chips .chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#route-profile-chips .chip-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.routeProfile = btn.dataset.profile;
    });
  });

  // Route Button
  document.getElementById('btn-compare-routes').addEventListener('click', executeRouteComparison);

  // Date input change
  document.getElementById('predict-date-input').addEventListener('change', (e) => {
    fetchPrediction(state.currentCoords.lat, state.currentCoords.lon, e.target.value);
  });

  // Geocoding search input
  const searchInput = document.getElementById('predict-search-input');
  const resultsDropdown = document.getElementById('predict-search-results');
  let debounceTimeout = null;

  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimeout);
    const q = searchInput.value.trim();
    if (q.length < 2) {
      resultsDropdown.style.display = 'none';
      return;
    }

    debounceTimeout = setTimeout(async () => {
      try {
        const resp = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
        const places = await resp.json();
        if (places && places.length > 0) {
          resultsDropdown.innerHTML = '';
          places.forEach(p => {
            const li = document.createElement('li');
            li.innerText = `${p.name}, ${p.admin1 || ''} (${p.country || ''})`;
            li.addEventListener('click', () => {
              searchInput.value = p.name;
              resultsDropdown.style.display = 'none';
              state.predictMap.setView([p.latitude, p.longitude], 9);
              fetchPrediction(p.latitude, p.longitude);
            });
            resultsDropdown.appendChild(li);
          });
          resultsDropdown.style.display = 'block';
        } else {
          resultsDropdown.style.display = 'none';
        }
      } catch (err) {
        console.error('Geocode search error:', err);
      }
    }, 300);
  });

  // Replay Dropdown
  document.getElementById('replay-events-select').addEventListener('change', (e) => {
    const val = e.target.value;
    if (!val) return;
    const parts = val.split(',');
    const lat = parseFloat(parts[0]);
    const lon = parseFloat(parts[1]);
    const dt = parts[2];

    document.getElementById('predict-date-input').value = dt;
    state.predictMap.setView([lat, lon], 9);
    fetchPrediction(lat, lon, dt);
  });
}

async function loadReplayEvents() {
  try {
    const resp = await fetch('/api/events/replay');
    const events = await resp.json();
    const select = document.getElementById('replay-events-select');
    select.innerHTML = '<option value="">Select a historical landslide to replay...</option>';

    events.forEach(ev => {
      const opt = document.createElement('option');
      opt.value = `${ev.latitude},${ev.longitude},${ev.date}`;
      opt.innerText = `${ev.date} — ${ev.description} (${ev.latitude.toFixed(2)}, ${ev.longitude.toFixed(2)})`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Error loading replay events:', err);
  }
}
