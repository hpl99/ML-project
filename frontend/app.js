/**
 * URsafe Frontend Application Logic
 * Integrates Leaflet Maps, Live Meteorological Telemetry, Chart.js,
 * Web Speech AI Companion Guidance, and Multi-Route Comparison.
 */

// Tile Layer Configurations
const TILE_SERVERS = {
  streets: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    options: { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }
  },
  topo: {
    url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    options: { attribution: '&copy; OpenTopoMap (CC-BY-SA)', maxZoom: 17 }
  },
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    options: { attribution: '&copy; Esri World Imagery', maxZoom: 18 }
  },
  dark: {
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
    options: { attribution: '&copy; CartoDB DarkMatter', maxZoom: 19 }
  }
};

// Band Color Mappings
const BAND_COLORS = {
  Green: '#10b981',
  Yellow: '#f59e0b',
  Orange: '#f97316',
  Red: '#ef4444'
};

const ARCHETYPE_COLORS = {
  fast: '#f97316',     // Amber / Orange
  balanced: '#0284c7', // Sky Blue
  safer: '#10b981'     // Emerald Green
};

// State Management
const state = {
  predictMap: null,
  predictMarker: null,
  predictTileLayer: null,
  currentCoords: { lat: 30.3244, lon: 78.0339 }, // Default: Dehradun, Uttarakhand
  predictProfile: 'hiker',
  currentPredictionData: null,

  // Route Section State
  routeMap: null,
  routeTileLayer: null,
  routeFeatureGroup: null,
  routeStartMarker: null,
  routeEndMarker: null,
  routeStartCoord: [30.0869, 78.2676], // Rishikesh
  routeEndCoord: [30.5574, 79.5667],   // Joshimath
  routeProfile: 'hiker',
  routeViewMode: 'single',             // 'single' (Clean, active only) | 'all'
  currentRouteData: null,
  selectedRouteIndex: 0,
  pickRouteMode: false,
  pickRouteStep: 'start',

  // Charts & Audio
  rainfallChart: null,
  routeElevationChart: null,
  isSpeaking: false,
  speechSynth: window.speechSynthesis || null
};

// SVG Pin Marker Factory
function createPinIcon(bandName) {
  const color = BAND_COLORS[bandName] || '#0f172a';
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 36" width="30" height="44">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 9 12 24 12 24s12-15 12-24c0-6.63-5.37-12-12-12z" fill="${color}" stroke="#ffffff" stroke-width="2"/>
      <circle cx="12" cy="12" r="5" fill="#ffffff"/>
    </svg>
  `;
  return L.divIcon({
    className: 'custom-pin-marker',
    html: svg,
    iconSize: [30, 44],
    iconAnchor: [15, 44],
    popupAnchor: [0, -40]
  });
}

function createEndpointPin(letter, bgHex) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 40" width="32" height="46">
      <path d="M14 0C6.27 0 0 6.27 0 14c0 10.5 14 26 14 26s14-15.5 14-26c0-7.73-6.27-14-14-14z" fill="${bgHex}" stroke="#ffffff" stroke-width="2"/>
      <circle cx="14" cy="14" r="8" fill="#ffffff"/>
      <text x="14" y="18" font-size="12" font-weight="bold" fill="${bgHex}" text-anchor="middle" font-family="sans-serif">${letter}</text>
    </svg>
  `;
  return L.divIcon({
    className: 'endpoint-pin-marker',
    html: svg,
    iconSize: [32, 46],
    iconAnchor: [16, 46],
    popupAnchor: [0, -42]
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

  // Initial prediction at default coordinate (Dehradun)
  fetchPrediction(state.currentCoords.lat, state.currentCoords.lon);
});

// -----------------------------------------------------------------------------
// 2. PREDICT SECTION LOGIC (Real-Time Weather & Terrain Risk)
// -----------------------------------------------------------------------------
function initPredictMap() {
  state.predictMap = L.map('predict-map', {
    center: [state.currentCoords.lat, state.currentCoords.lon],
    zoom: 8,
    zoomControl: true
  });

  state.predictTileLayer = L.tileLayer(TILE_SERVERS.streets.url, TILE_SERVERS.streets.options).addTo(state.predictMap);

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

  // Place/update map marker immediately
  if (state.predictMarker) {
    state.predictMap.removeLayer(state.predictMarker);
  }
  state.predictMarker = L.marker([lat, lon], {
    icon: createPinIcon('Green')
  }).addTo(state.predictMap);

  // Show loading indicator in location bar
  const locNameElem = document.getElementById('res-location-name');
  if (locNameElem) locNameElem.innerText = 'Scanning satellite & live weather telemetry...';

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
    state.currentPredictionData = data;
    renderPredictionResults(data);

    // Update pin color by hazard band
    state.predictMarker.setIcon(createPinIcon(data.band));
    const locDisplay = data.location_name || `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
    state.predictMarker.bindPopup(`
      <div style="font-family: var(--font-body); padding: 4px;">
        <strong style="font-size: 14px;">${data.band_label}</strong> (${data.score}/100)<br>
        <span style="font-size: 12px; color: #475569;">${locDisplay}</span><br>
        <span style="font-size: 11px; color: #0284c7;">${data.current_weather?.weather_icon || '☀️'} ${data.current_weather?.temperature_c || 20}°C &bull; ${data.current_weather?.weather_description || 'Clear'}</span>
      </div>
    `).openPopup();
  } catch (err) {
    console.error('Error fetching prediction:', err);
    showToast('Failed to retrieve live hazard prediction. Please verify server connection.');
  }
}

function renderPredictionResults(data) {
  document.getElementById('predict-empty-state').style.display = 'none';
  const card = document.getElementById('predict-card');
  card.style.display = 'block';

  // 1. Location Bar
  const locName = data.location_name || `${data.coordinates.lat.toFixed(3)}°N, ${data.coordinates.lon.toFixed(3)}°E`;
  document.getElementById('res-location-name').innerText = locName;
  const elev = data.features?.elevation_m || data.terrain_stability?.elevation_m || 0;
  document.getElementById('res-coords-elev').innerText = `${data.coordinates.lat.toFixed(4)}°N, ${data.coordinates.lon.toFixed(4)}°E • Elevation ${elev} m a.s.l.`;

  // 2. Score & Band
  document.getElementById('res-score').innerText = data.score;
  document.getElementById('res-confidence').innerText = data.confidence;

  const bandBadge = document.getElementById('res-band-badge');
  bandBadge.className = `band-pill-badge band-${data.band}`;
  bandBadge.innerText = `${data.band} (${data.band_label})`;

  // 3. Real-Time Meteorological Telemetry Widget
  const w = data.current_weather || {};
  document.getElementById('res-weather-icon').innerText = w.weather_icon || '🌤️';
  document.getElementById('res-weather-temp').innerText = w.temperature_c !== undefined ? w.temperature_c : '--';
  document.getElementById('res-weather-desc').innerText = w.weather_description || 'Fair';
  document.getElementById('res-weather-apparent').innerText = `${w.apparent_temperature_c || w.temperature_c || '--'}°C`;
  document.getElementById('res-weather-precip').innerText = `${w.precipitation_mm || 0.0} mm/h`;
  document.getElementById('res-weather-humidity').innerText = `${w.relative_humidity_pct || '--'}%`;
  document.getElementById('res-weather-wind').innerText = `${w.wind_speed_kmh || 0} km/h`;
  document.getElementById('res-weather-pressure').innerText = `${w.surface_pressure_hpa || 1013} hPa`;
  document.getElementById('res-weather-clouds').innerText = `${w.cloud_cover_pct || 0}%`;
  document.getElementById('res-weather-gusts').innerText = `${w.wind_gusts_kmh || 0} km/h`;
  document.getElementById('res-weather-time').innerText = w.observation_time ? `Live: ${w.observation_time}` : 'Open-Meteo Realtime';

  // 4. Hydrometeorological Saturation & Slope Stability Breakdown
  const hydro = data.hydrology_status || {};
  const satPct = hydro.soil_saturation_pct || 25;
  document.getElementById('res-soil-saturation-val').innerText = `${satPct}%`;
  const satBar = document.getElementById('res-soil-saturation-bar');
  satBar.style.width = `${Math.min(satPct, 100)}%`;
  satBar.className = `progress-fill ${satPct > 75 ? 'fill-red' : (satPct > 55 ? 'fill-orange' : (satPct > 35 ? 'fill-yellow' : 'fill-green'))}`;
  document.getElementById('res-soil-saturation-label').innerText = hydro.soil_status_label || 'Moderate Saturation';

  const terrain = data.terrain_stability || {};
  const slopeDeg = terrain.slope_deg || data.features?.slope_deg || 5;
  document.getElementById('res-slope-gradient-val').innerText = `${slopeDeg}°`;
  const slopeBar = document.getElementById('res-slope-gradient-bar');
  const slopePct = Math.min((slopeDeg / 45.0) * 100.0, 100);
  slopeBar.style.width = `${slopePct}%`;
  slopeBar.className = `progress-fill ${slopeDeg > 35 ? 'fill-red' : (slopeDeg > 25 ? 'fill-orange' : (slopeDeg > 15 ? 'fill-yellow' : 'fill-green'))}`;
  document.getElementById('res-slope-gradient-label').innerText = `${terrain.slope_category || 'Mountain Slope'} • ${terrain.stability_rating || ''}`;

  document.getElementById('res-water-dist').innerText = `${Math.round(data.features?.distance_to_water_m || 5000)} m`;
  document.getElementById('res-api-val').innerText = `${(data.features?.api_30d || 0).toFixed(1)} mm`;

  // 5. AI Companion Guidance
  document.getElementById('res-companion-text').innerText = data.advice;

  // 6. Terrain Zone
  document.getElementById('res-zone-name').innerText = data.terrain_zone?.name || 'Mountain Pass';
  const membershipsDiv = document.getElementById('res-zone-memberships');
  membershipsDiv.innerHTML = '';
  if (data.terrain_zone?.memberships) {
    for (const [zName, prob] of Object.entries(data.terrain_zone.memberships)) {
      const span = document.createElement('span');
      span.innerText = `${zName}: ${(prob * 100).toFixed(0)}%`;
      membershipsDiv.appendChild(span);
    }
  }

  // 7. Top SHAP Factors List
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

  // 8. Freshness
  document.getElementById('res-freshness').innerText = `${data.date} &bull; Model ${data.model_version}`;

  // 9. Rainfall Chart
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
// 3. ROUTE COMPARISON & NAVIGATION LOGIC (Clean Map Without Lingering Routes)
// -----------------------------------------------------------------------------
function initRouteMap() {
  state.routeMap = L.map('route-map', {
    center: [30.3, 78.9],
    zoom: 8,
    zoomControl: true
  });

  state.routeTileLayer = L.tileLayer(TILE_SERVERS.streets.url, TILE_SERVERS.streets.options).addTo(state.routeMap);

  // Dedicated feature group to manage route lines & markers atomically
  state.routeFeatureGroup = L.featureGroup().addTo(state.routeMap);

  // Map Click Listener for interactive origin/destination picking
  state.routeMap.on('click', async (e) => {
    if (!state.pickRouteMode) return;
    const { lat, lng } = e.latlng;

    if (state.pickRouteStep === 'start') {
      state.routeStartCoord = [lat, lng];
      const place = await reverseGeocodeCoord(lat, lng);
      document.getElementById('route-start-input').value = place;
      state.pickRouteStep = 'end';
      document.getElementById('route-instruction-text').innerHTML = 'Origin placed! Now click destination point for <strong>Destination (B)</strong>...';
      renderEndpointMarkersOnly();
    } else {
      state.routeEndCoord = [lat, lng];
      const place = await reverseGeocodeCoord(lat, lng);
      document.getElementById('route-end-input').value = place;
      state.pickRouteMode = false;
      document.getElementById('route-map-instruction').style.display = 'none';
      document.getElementById('btn-pick-endpoints-map').classList.remove('active');
      renderEndpointMarkersOnly();
      executeRouteComparison();
    }
  });
}

async function reverseGeocodeCoord(lat, lon) {
  try {
    const resp = await fetch(`/api/reverse-geocode?lat=${lat}&lon=${lon}`);
    if (resp.ok) {
      const data = await resp.json();
      return data.place_name || `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
    }
  } catch (e) {
    console.warn('Reverse geocode failed:', e);
  }
  return `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
}

async function geocodePlaceName(name) {
  try {
    const resp = await fetch(`/api/geocode?q=${encodeURIComponent(name)}`);
    if (resp.ok) {
      const places = await resp.json();
      if (places && places.length > 0) {
        return [places[0].latitude, places[0].longitude];
      }
    }
  } catch (e) {
    console.warn('Geocoding place failed:', e);
  }
  return null;
}

function clearRouteMapLayers() {
  // 1. Clear atomic feature group
  if (state.routeFeatureGroup) {
    state.routeFeatureGroup.clearLayers();
  }

  // 2. Extra safety: Purge any lingering polyline or vector path on the map
  state.routeMap.eachLayer((layer) => {
    if ((layer instanceof L.Polyline || layer instanceof L.Marker) && layer !== state.routeTileLayer) {
      state.routeMap.removeLayer(layer);
    }
  });
}

function renderEndpointMarkersOnly() {
  clearRouteMapLayers();

  if (state.routeStartCoord) {
    const startPin = L.marker(state.routeStartCoord, {
      icon: createEndpointPin('A', '#10b981')
    }).bindPopup('<strong>Origin (A)</strong>');
    state.routeFeatureGroup.addLayer(startPin);
  }

  if (state.routeEndCoord) {
    const endPin = L.marker(state.routeEndCoord, {
      icon: createEndpointPin('B', '#ef4444')
    }).bindPopup('<strong>Destination (B)</strong>');
    state.routeFeatureGroup.addLayer(endPin);
  }
}

async function executeRouteComparison() {
  const btn = document.getElementById('btn-compare-routes');
  const loading = document.getElementById('route-loading');
  const cardsList = document.getElementById('route-cards-list');

  btn.disabled = true;
  loading.style.display = 'block';
  cardsList.innerHTML = '';
  document.getElementById('route-chart-container').style.display = 'none';
  document.getElementById('route-breakdown-container').style.display = 'none';

  // 1. Clear ANY existing route layers completely to prevent orange or old routes lingering
  clearRouteMapLayers();

  try {
    const startInputVal = document.getElementById('route-start-input').value.trim();
    const endInputVal = document.getElementById('route-end-input').value.trim();

    // Check if geocoding needed for text inputs
    if (startInputVal && isNaN(startInputVal.split(',')[0])) {
      const resolved = await geocodePlaceName(startInputVal);
      if (resolved) state.routeStartCoord = resolved;
    }
    if (endInputVal && isNaN(endInputVal.split(',')[0])) {
      const resolved = await geocodePlaceName(endInputVal);
      if (resolved) state.routeEndCoord = resolved;
    }

    const resp = await fetch('/api/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start: state.routeStartCoord,
        end: state.routeEndCoord,
        profile: state.routeProfile
      })
    });

    const data = await resp.json();
    loading.style.display = 'none';
    btn.disabled = false;

    if (data.routes && data.routes.length > 0) {
      state.currentRouteData = data;
      state.selectedRouteIndex = 0; // Default to first (or safer)

      // Render weather telemetry at route endpoints
      renderRouteEndpointsWeather(data);

      // Render route comparison cards
      renderRouteCards(data.routes);

      // Draw route on map cleanly without old lingering layers
      drawSelectedRoute();
    } else {
      cardsList.innerHTML = '<div style="color: var(--color-band-orange); padding: 14px; background: rgba(249, 115, 22, 0.1); border-radius: 8px;">No navigable driving routes found between these locations. Try nearby highway junctions.</div>';
    }
  } catch (err) {
    loading.style.display = 'none';
    btn.disabled = false;
    console.error('Error calculating routes:', err);
    cardsList.innerHTML = '<div class="error-msg" style="color: var(--color-band-orange); padding: 14px; background: rgba(249, 115, 22, 0.1); border-radius: 8px; font-size: 13px;">Route calculation temporarily unavailable. Please retry.</div>';
  }
}

function renderRouteEndpointsWeather(data) {
  const bar = document.getElementById('route-endpoints-weather-bar');
  if (!bar) return;

  const orig = data.origin || {};
  const dest = data.destination || {};

  document.getElementById('route-origin-name').innerText = orig.name || 'Origin';
  document.getElementById('route-dest-name').innerText = dest.name || 'Destination';

  const origW = orig.weather || {};
  const destW = dest.weather || {};

  document.getElementById('route-origin-weather').innerText = `${origW.weather_icon || '🌤️'} ${origW.temperature_c || 20}°C &bull; ${origW.weather_description || 'Fair'} (${origW.precipitation_mm || 0} mm/h)`;
  document.getElementById('route-dest-weather').innerText = `${destW.weather_icon || '🌤️'} ${destW.temperature_c || 18}°C &bull; ${destW.weather_description || 'Fair'} (${destW.precipitation_mm || 0} mm/h)`;

  bar.style.display = 'flex';
}

function renderRouteCards(routes) {
  const container = document.getElementById('route-cards-list');
  container.innerHTML = '';

  routes.forEach((r, idx) => {
    const card = document.createElement('div');
    const isSelected = idx === state.selectedRouteIndex;
    card.className = `route-card ${isSelected ? 'selected' : ''}`;
    card.id = `route-card-${r.id}`;

    // Archetype badge color
    const archColor = r.archetype_color || ARCHETYPE_COLORS[r.id] || '#0f172a';

    card.innerHTML = `
      <div class="route-card-header">
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="route-archetype" style="color: ${archColor}; font-weight: 700;">${r.archetype}</span>
          <span style="font-size: 11px; padding: 2px 8px; border-radius: 9999px; background: #f1f5f9; color: var(--color-primary-text);">${r.badge}</span>
        </div>
        <div style="font-weight: 700; font-size: 15px;">${r.duration_min} min</div>
      </div>
      <div style="font-size: 13px; color: var(--color-muted-text); margin: 6px 0;">${r.why}</div>
      <div class="route-stats-grid">
        <div>
          <span style="color: var(--color-muted-text); font-size: 11px;">Distance</span><br>
          <span class="route-stat-val">${r.distance_km} km</span>
        </div>
        <div>
          <span style="color: var(--color-muted-text); font-size: 11px;">Mean Hazard</span><br>
          <span class="route-stat-val">${r.mean_risk}/100</span>
        </div>
        <div>
          <span style="color: var(--color-muted-text); font-size: 11px;">Peak Hazard</span><br>
          <span class="route-stat-val" style="color: ${BAND_COLORS[r.max_risk > 75 ? 'Red' : (r.max_risk > 50 ? 'Orange' : 'Green')]};">${r.max_risk}/100</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      document.querySelectorAll('.route-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      state.selectedRouteIndex = idx;
      drawSelectedRoute();
    });

    container.appendChild(card);
  });
}

/**
 * Draws the selected route on the map.
 * CRITICAL FIX: Clears all previous layers completely so NO old orange routes linger when switching to green!
 */
function drawSelectedRoute() {
  if (!state.currentRouteData || !state.currentRouteData.routes) return;
  const routes = state.currentRouteData.routes;
  const selectedRoute = routes[state.selectedRouteIndex];
  if (!selectedRoute) return;

  // 1. Purge EVERYTHING currently drawn on the route map
  clearRouteMapLayers();

  const bounds = L.latLngBounds();

  // 2. If 'all' mode: Draw inactive alternative routes in subtle light gray first
  if (state.routeViewMode === 'all') {
    routes.forEach((r, idx) => {
      if (idx !== state.selectedRouteIndex) {
        (r.segments || []).forEach(seg => {
          const coords = seg.geometry.coordinates.map(c => [c[1], c[0]]);
          const polyline = L.polyline(coords, {
            color: '#94a3b8',
            weight: 4,
            opacity: 0.45,
            dashArray: '4, 8'
          });
          state.routeFeatureGroup.addLayer(polyline);
        });
      }
    });
  }

  // 3. Draw Active Route Prominently
  // Signature archetype color for consistent clean aesthetic
  const archColor = selectedRoute.archetype_color || ARCHETYPE_COLORS[selectedRoute.id] || '#10b981';

  // Base background casing for high visibility
  (selectedRoute.segments || []).forEach(seg => {
    const coords = seg.geometry.coordinates.map(c => [c[1], c[0]]);
    const casing = L.polyline(coords, {
      color: '#ffffff',
      weight: 8,
      opacity: 0.95
    });
    state.routeFeatureGroup.addLayer(casing);
    coords.forEach(c => bounds.extend(c));
  });

  // Active route polylines:
  // For the Safer route (id: 'safer' or archetype: 'Safer'), use clean Green (#10b981) without any lingering orange!
  // For other routes, segment coloring or archetype coloring is cleanly used.
  (selectedRoute.segments || []).forEach((seg, sIdx) => {
    let segmentColor = archColor;
    if (selectedRoute.id === 'safer') {
      segmentColor = '#10b981'; // Guaranteed clean emerald green for Safer route
    } else if (selectedRoute.id === 'fast') {
      segmentColor = '#f97316'; // Clean amber/orange for Fast route
    } else {
      segmentColor = BAND_COLORS[seg.properties.band] || archColor;
    }

    const coords = seg.geometry.coordinates.map(c => [c[1], c[0]]);
    const polyline = L.polyline(coords, {
      color: segmentColor,
      weight: 6,
      opacity: 0.95
    });

    polyline.bindPopup(`
      <div style="font-family: var(--font-body); padding: 4px;">
        <strong>Segment ${sIdx + 1} (${selectedRoute.archetype})</strong><br>
        Hazard Score: <strong>${seg.properties.score}/100</strong> (${seg.properties.band})<br>
        Slope: ${seg.properties.slope_deg}&deg; &bull; 3-Day Rain: ${seg.properties.rain_3d} mm
      </div>
    `);

    state.routeFeatureGroup.addLayer(polyline);
  });

  // 4. Place Origin ('A') and Destination ('B') Pins
  if (selectedRoute.segments && selectedRoute.segments.length > 0) {
    const firstSeg = selectedRoute.segments[0];
    const lastSeg = selectedRoute.segments[selectedRoute.segments.length - 1];

    const startCoord = [firstSeg.geometry.coordinates[0][1], firstSeg.geometry.coordinates[0][0]];
    const endCoord = [lastSeg.geometry.coordinates[lastSeg.geometry.coordinates.length - 1][1], lastSeg.geometry.coordinates[lastSeg.geometry.coordinates.length - 1][0]];

    const startPin = L.marker(startCoord, {
      icon: createEndpointPin('A', '#10b981'),
      zIndexOffset: 1000
    }).bindPopup(`<strong>Origin:</strong> ${document.getElementById('route-start-input').value}`);

    const endPin = L.marker(endCoord, {
      icon: createEndpointPin('B', '#ef4444'),
      zIndexOffset: 1000
    }).bindPopup(`<strong>Destination:</strong> ${document.getElementById('route-end-input').value}`);

    state.routeFeatureGroup.addLayer(startPin);
    state.routeFeatureGroup.addLayer(endPin);
  }

  // 5. Fit Bounds smoothly
  if (bounds.isValid()) {
    state.routeMap.fitBounds(bounds, { padding: [50, 50] });
  }

  // 6. Update Route Elevation Profile Chart
  renderRouteElevationChart(selectedRoute);

  // 7. Update Segment Breakdown Table
  renderRouteSegmentsTable(selectedRoute);
}

function renderRouteElevationChart(routeObj) {
  const chartBox = document.getElementById('route-chart-container');
  chartBox.style.display = 'block';

  const canvas = document.getElementById('route-elevation-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  if (state.routeElevationChart) {
    state.routeElevationChart.destroy();
  }

  const segments = routeObj.segments || [];
  if (segments.length === 0) return;

  const totalDist = routeObj.distance_km || 100;
  const distPerSeg = totalDist / segments.length;

  const labels = [];
  const elevationData = [];
  const pointColors = [];

  // Approximate elevation along segments
  let baseElev = 400.0;
  segments.forEach((seg, idx) => {
    const km = (idx * distPerSeg).toFixed(1);
    labels.push(`${km} km`);
    // Estimate elevation progression based on slope
    const slope = seg.properties.slope_deg || 5;
    baseElev += (slope * 25.0) * (idx < segments.length / 2 ? 1 : 0.5);
    elevationData.push(Math.round(baseElev));
    pointColors.push(BAND_COLORS[seg.properties.band] || '#3b82f6');
  });

  const archColor = routeObj.archetype_color || ARCHETYPE_COLORS[routeObj.id] || '#10b981';

  state.routeElevationChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Elevation Profile (m)',
        data: elevationData,
        borderColor: archColor,
        backgroundColor: `${archColor}22`,
        fill: true,
        tension: 0.3,
        pointBackgroundColor: pointColors,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Waypoint at ${items[0].label}`,
            label: (item) => {
              const seg = segments[item.dataIndex];
              return `Elevation: ${item.raw}m | Hazard: ${seg?.properties.score}/100 (${seg?.properties.band})`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { font: { size: 10 }, maxTicksLimit: 8 }
        },
        y: {
          grid: { color: 'rgba(0,0,0,0.05)' },
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
}

function renderRouteSegmentsTable(routeObj) {
  const container = document.getElementById('route-breakdown-container');
  const tableDiv = document.getElementById('route-segments-table');
  container.style.display = 'block';

  tableDiv.innerHTML = '';
  const segments = routeObj.segments || [];

  segments.forEach((seg, idx) => {
    const row = document.createElement('div');
    row.className = 'segment-item-row';
    const band = seg.properties.band;
    const bandColor = BAND_COLORS[band] || '#10b981';

    row.innerHTML = `
      <div>
        <strong>Segment ${idx + 1}</strong> (~${((idx + 1) * 2).toFixed(0)} km)
        <span style="color: var(--color-muted-text); font-size: 11px; margin-left: 6px;">Slope: ${seg.properties.slope_deg}&deg;</span>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <span style="color: var(--color-muted-text); font-size: 11px;">Rain 3d: ${seg.properties.rain_3d} mm</span>
        <span class="band-pill-badge band-${band}" style="padding: 2px 8px; font-size: 10px; background: ${bandColor}20; color: ${bandColor}; border: 1px solid ${bandColor}40;">
          ${seg.properties.score}/100 (${band})
        </span>
      </div>
    `;
    tableDiv.appendChild(row);
  });
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
// 5. EVENT LISTENERS, SPEECH SYNTHESIS & UI CONTROLS
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

  // Quick Preset Location Chips for Predict Map
  document.querySelectorAll('.preset-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const lat = parseFloat(btn.dataset.lat);
      const lon = parseFloat(btn.dataset.lon);
      state.predictMap.flyTo([lat, lon], 10, { duration: 1.2 });
      fetchPrediction(lat, lon);
    });
  });

  // Quick Preset Corridors for Route Map
  document.querySelectorAll('.route-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('route-start-input').value = btn.dataset.start;
      document.getElementById('route-end-input').value = btn.dataset.end;
      const s = btn.dataset.startCoord.split(',').map(Number);
      const e = btn.dataset.endCoord.split(',').map(Number);
      state.routeStartCoord = s;
      state.routeEndCoord = e;
      executeRouteComparison();
    });
  });

  // GPS Locate Me Button
  const locateBtn = document.getElementById('btn-locate-me');
  if (locateBtn) {
    locateBtn.addEventListener('click', () => {
      if (!navigator.geolocation) {
        showToast('Geolocation is not supported by your browser.');
        return;
      }
      locateBtn.innerHTML = '<span>Locating...</span>';
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          locateBtn.innerHTML = '<span>Locate Me</span>';
          state.predictMap.flyTo([lat, lon], 11, { duration: 1.5 });
          fetchPrediction(lat, lon);
          showToast(`Located GPS: ${lat.toFixed(3)}, ${lon.toFixed(3)}`);
        },
        (err) => {
          locateBtn.innerHTML = '<span>Locate Me</span>';
          console.warn('Geolocation error:', err);
          showToast('Unable to access device location. Please search manually.');
        },
        { timeout: 8000 }
      );
    });
  }

  // Layer Selectors
  document.getElementById('predict-layer-select')?.addEventListener('change', (e) => {
    const key = e.target.value;
    if (state.predictTileLayer) state.predictMap.removeLayer(state.predictTileLayer);
    state.predictTileLayer = L.tileLayer(TILE_SERVERS[key].url, TILE_SERVERS[key].options).addTo(state.predictMap);
  });

  document.getElementById('route-layer-select')?.addEventListener('change', (e) => {
    const key = e.target.value;
    if (state.routeTileLayer) state.routeMap.removeLayer(state.routeTileLayer);
    state.routeTileLayer = L.tileLayer(TILE_SERVERS[key].url, TILE_SERVERS[key].options).addTo(state.routeMap);
  });

  // Route Button
  document.getElementById('btn-compare-routes').addEventListener('click', executeRouteComparison);

  // Swap Route Endpoints Button
  document.getElementById('btn-swap-endpoints')?.addEventListener('click', () => {
    const startIn = document.getElementById('route-start-input');
    const endIn = document.getElementById('route-end-input');
    const tempVal = startIn.value;
    startIn.value = endIn.value;
    endIn.value = tempVal;

    const tempCoord = state.routeStartCoord;
    state.routeStartCoord = state.routeEndCoord;
    state.routeEndCoord = tempCoord;

    showToast('Swapped route origin and destination.');
  });

  // Pick Endpoints on Map Toggle
  const pickBtn = document.getElementById('btn-pick-endpoints-map');
  const banner = document.getElementById('route-map-instruction');
  pickBtn?.addEventListener('click', () => {
    state.pickRouteMode = !state.pickRouteMode;
    state.pickRouteStep = 'start';
    if (state.pickRouteMode) {
      pickBtn.classList.add('active');
      banner.style.display = 'flex';
      document.getElementById('route-instruction-text').innerHTML = 'Click on the map to place the <strong>Origin (Start)</strong> pin...';
    } else {
      pickBtn.classList.remove('active');
      banner.style.display = 'none';
    }
  });

  document.getElementById('btn-cancel-route-pick')?.addEventListener('click', () => {
    state.pickRouteMode = false;
    pickBtn.classList.remove('active');
    banner.style.display = 'none';
  });

  // Clear Route Map Button
  document.getElementById('btn-clear-route')?.addEventListener('click', () => {
    clearRouteMapLayers();
    document.getElementById('route-cards-list').innerHTML = '';
    document.getElementById('route-chart-container').style.display = 'none';
    document.getElementById('route-breakdown-container').style.display = 'none';
    document.getElementById('route-endpoints-weather-bar').style.display = 'none';
    showToast('Cleared route corridor from map.');
  });

  // Route View Mode Radio Toggle (Single vs All)
  document.querySelectorAll('input[name="route-view-mode"]').forEach(r => {
    r.addEventListener('change', (e) => {
      state.routeViewMode = e.target.value;
      drawSelectedRoute();
    });
  });

  // Accordion toggle for segments breakdown
  document.getElementById('btn-toggle-segments')?.addEventListener('click', () => {
    const tbl = document.getElementById('route-segments-table');
    const arrow = document.getElementById('accordion-arrow');
    if (tbl.style.display === 'none') {
      tbl.style.display = 'block';
      arrow.innerText = '▲';
    } else {
      tbl.style.display = 'none';
      arrow.innerText = '▼';
    }
  });

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
              state.predictMap.flyTo([p.latitude, p.longitude], 10, { duration: 1.2 });
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

  // Close dropdown on click outside
  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !resultsDropdown.contains(e.target)) {
      resultsDropdown.style.display = 'none';
    }
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
    state.predictMap.flyTo([lat, lon], 10, { duration: 1.2 });
    fetchPrediction(lat, lon, dt);
  });

  // Voice AI Companion Guidance (Web Speech API)
  const voiceBtn = document.getElementById('btn-voice-guidance');
  const waveform = document.getElementById('companion-audio-waveform');
  if (voiceBtn) {
    voiceBtn.addEventListener('click', () => {
      if (!state.speechSynth) {
        showToast('Speech synthesis is not supported on this browser.');
        return;
      }

      if (state.isSpeaking) {
        state.speechSynth.cancel();
        state.isSpeaking = false;
        voiceBtn.classList.remove('speaking');
        document.getElementById('voice-icon').innerText = '🔊';
        document.getElementById('voice-text').innerText = 'Listen';
        waveform.style.display = 'none';
        return;
      }

      const text = document.getElementById('res-companion-text').innerText;
      if (!text) return;

      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1.0;
      utter.pitch = 1.0;

      utter.onstart = () => {
        state.isSpeaking = true;
        voiceBtn.classList.add('speaking');
        document.getElementById('voice-icon').innerText = '⏹️';
        document.getElementById('voice-text').innerText = 'Stop';
        waveform.style.display = 'flex';
      };

      utter.onend = () => {
        state.isSpeaking = false;
        voiceBtn.classList.remove('speaking');
        document.getElementById('voice-icon').innerText = '🔊';
        document.getElementById('voice-text').innerText = 'Listen';
        waveform.style.display = 'none';
      };

      utter.onerror = () => {
        state.isSpeaking = false;
        voiceBtn.classList.remove('speaking');
        document.getElementById('voice-icon').innerText = '🔊';
        document.getElementById('voice-text').innerText = 'Listen';
        waveform.style.display = 'none';
      };

      state.speechSynth.speak(utter);
    });
  }

  // Copy Advice to Clipboard
  document.getElementById('btn-copy-advice')?.addEventListener('click', () => {
    const text = document.getElementById('res-companion-text').innerText;
    if (text) {
      navigator.clipboard.writeText(text);
      showToast('AI Companion advisory copied to clipboard!');
    }
  });

  // Emergency SOS Modal
  const emergModal = document.getElementById('emergency-modal');
  document.getElementById('btn-open-emergency')?.addEventListener('click', () => {
    emergModal.style.display = 'flex';
  });
  document.getElementById('btn-close-emergency')?.addEventListener('click', () => {
    emergModal.style.display = 'none';
  });
  emergModal?.addEventListener('click', (e) => {
    if (e.target === emergModal) emergModal.style.display = 'none';
  });
}

function showToast(message) {
  const toast = document.getElementById('app-toast');
  if (!toast) return;
  toast.innerText = message;
  toast.style.display = 'block';
  setTimeout(() => {
    toast.style.display = 'none';
  }, 3500);
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
