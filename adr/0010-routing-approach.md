# ADR 0010: Multi-Criteria Routing, Segmentation, and Risk Aggregation

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Travelers need to compare alternative travel routes between origins and destinations in mountainous terrain. Standard navigation engines (Google Maps, OSRM) optimize exclusively for travel time or shortest distance, frequently directing drivers through highly unstable canyon roads during monsoon downpours. Section 6 and Section 7 require generating up to 3 alternative routes from OSRM, segmenting them by risk, and computing route-level safety trade-offs.

## Decision
1. **Upstream Routing**:
   - Query OSRM public demo router: `https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?alternatives=true&overview=full&geometries=geojson`.
   - Respect driving profile limitations of public OSRM. If fewer than 3 distinct routes are returned by OSRM, state so honestly in the API response and UI.
2. **Waypoints & Spatial Sampling**:
   - Sample route coordinates approximately every 2.0 km (capped at 30 sample points per route to preserve sub-20s response times).
   - Batch elevation and weather queries; reuse the 0.25° grid cache for identical regions.
3. **Route Risk Heuristics**:
   - For each sampled segment $i$ with risk probability $p_i = \text{score}_i / 100$:
     - Segment Risk Score: $\text{score}_i \in [0, 100]$ mapped to Green/Yellow/Orange/Red.
     - Mean Route Risk: $\frac{1}{N} \sum_{i=1}^N \text{score}_i$.
     - Maximum Segment Risk: $\max_i(\text{score}_i)$.
     - Cumulative Route Risk Heuristic: $R_{\text{route}} = 1 - \prod_{i=1}^N (1 - p_i)$. This metric is clearly labeled in the UI as an independence heuristic.
4. **Three Route Archetypes**:
   - **Fast**: Minimal travel time.
   - **Safer**: Minimal maximum segment risk $\min(\max(\text{score}))$.
   - **Balanced**: Lowest composite objective $J = \alpha \cdot \frac{T}{T_{\min}} + (1 - \alpha) \cdot \frac{R_{\text{mean}}}{R_{\max}}$, where weighting $\alpha$ adjusts by traveler profile.
5. **Color-Coded Geometry**:
   - Return GeoJSON MultiLineString segments individually styled by hazard band for map rendering.

## Consequences
- Enables drivers to make informed safety vs. time trade-offs (e.g. taking an extra 25 minutes to avoid a high-hazard canyon pass).
