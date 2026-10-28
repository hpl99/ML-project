"""AI Companion Guidance Module.

Generates transparent, personalized, and actionable plain-language safety advice
integrating traveler profiles, model confidence, top SHAP factors, and risk bands (ADR 0009).
Shared verbatim between the backend API and Jupyter Viva Practicals.
"""
import logging
from typing import Dict, Any, List

from ml.config import PROFILE_MULTIPLIERS, RISK_BANDS
from ml.feature_text import explain_feature_impact

logger = logging.getLogger(__name__)


def compute_confidence(feature_dict: Dict[str, Any]) -> str:
    """Evaluate data completeness and provenance to determine confidence score."""
    missing_count = sum(1 for v in feature_dict.values() if v is None or (isinstance(v, float) and v != v))
    soil_prov = feature_dict.get("soil_provenance", "observed")

    if missing_count >= 3 or soil_prov == "missing":
        return "Low"
    elif missing_count >= 1 or soil_prov == "derived":
        return "Medium"
    else:
        return "High"


def determine_risk_band(score: int) -> Dict[str, Any]:
    """Map calibrated integer score (0-100) to operational risk band."""
    for band_name, band_info in RISK_BANDS.items():
        if band_info["min"] <= score <= band_info["max"]:
            return {
                "name": band_name,
                "label": band_info["label"],
                "action": band_info["action"]
            }
    if score > 100:
        return {"name": "Red", "label": "Severe Hazard", "action": RISK_BANDS["Red"]["action"]}
    return {"name": "Green", "label": "Low Risk", "action": RISK_BANDS["Green"]["action"]}


def ai_companion_guide(
    terrain_sample: Dict[str, Any],
    profile: str = "hiker",
    raw_probability: float = None,
    shap_contributions: Dict[str, float] = None,
    terrain_zone: Dict[str, Any] = None,
    live_weather: Dict[str, Any] = None,
    location_name: str = None
) -> Dict[str, Any]:
    """Universal AI Companion guide callable by both the FastAPI backend and Jupyter notebooks.

    Parameters
    ----------
    terrain_sample : Dict[str, Any]
        Dictionary of terrain, weather, and hydrological measurements.
    profile : str
        Traveler persona ('hiker', 'rider', 'family', 'elderly').
    raw_probability : float, optional
        Calibrated model output probability [0.0, 1.0].
    shap_contributions : Dict[str, float], optional
        Feature-level SHAP values.
    terrain_zone : Dict[str, Any], optional
        Terrain classification zone and soft GMM membership probabilities.
    live_weather : Dict[str, Any], optional
        Real-time current meteorological observations from Open-Meteo.
    location_name : str, optional
        Reverse-geocoded place or landmark name.
    """
    # 1. Base Score Calculation
    if raw_probability is not None:
        base_score = int(round(raw_probability * 100.0))
    else:
        # Heuristic fallback if model not loaded
        slope = terrain_sample.get("slope_deg", 5.0)
        rain3d = terrain_sample.get("rain_3d", 10.0)
        base_score = int(min(max((slope * 1.5) + (rain3d * 0.4), 5.0), 95.0))

    # 2. Traveler Profile Multiplier (Section 5.8)
    profile_clean = profile.lower() if profile else "hiker"
    multiplier = PROFILE_MULTIPLIERS.get(profile_clean, 1.0)
    adjusted_score = int(min(round(base_score * multiplier), 100))

    # 3. Operational Risk Band
    band = determine_risk_band(adjusted_score)

    # 4. Confidence Evaluation
    confidence = compute_confidence(terrain_sample)

    # 5. Top Contributing Physical Factors
    top_factors: List[Dict[str, Any]] = []
    if shap_contributions:
        # Sort by absolute SHAP impact
        sorted_feats = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        for feat_name, s_val in sorted_feats[:5]:
            val = terrain_sample.get(feat_name, 0.0)
            prov = "derived" if feat_name.startswith("soil") and terrain_sample.get("soil_provenance") == "derived" else "observed"
            top_factors.append(explain_feature_impact(feat_name, val, s_val, provenance=prov))
    else:
        # Heuristic top factors based on domain thresholds
        slope = terrain_sample.get("slope_deg", 0.0)
        rain_3d = terrain_sample.get("rain_3d", 0.0)
        topsoil = terrain_sample.get("soil_wetness_top", 0.0)
        top_factors.append(explain_feature_impact("slope_deg", slope, shap_val=0.15 if slope > 20 else -0.1))
        top_factors.append(explain_feature_impact("rain_3d", rain_3d, shap_val=0.25 if rain_3d > 30 else -0.05))
        top_factors.append(explain_feature_impact("soil_wetness_top", topsoil, shap_val=0.1 if topsoil > 0.7 else -0.02))

    # 6. Companion Plain-Language Advice Synthesis
    slope_val = terrain_sample.get("slope_deg", 0.0)
    rain_3d_val = terrain_sample.get("rain_3d", 0.0)
    elev_val = terrain_sample.get("elevation_m", 0.0)

    # Live atmospheric telemetry synthesis
    loc_prefix = f"At {location_name} (elev. {elev_val:.0f}m): " if location_name else f"At elevation {elev_val:.0f}m: "
    weather_snippet = ""
    if live_weather:
        w_desc = live_weather.get("weather_description", "Fair")
        w_temp = live_weather.get("temperature_c", 20.0)
        w_precip = live_weather.get("precipitation_mm", 0.0)
        w_wind = live_weather.get("wind_speed_kmh", 0.0)
        w_hum = live_weather.get("relative_humidity_pct", 50)
        if w_precip > 0.5:
            weather_snippet = f"Current condition: {w_desc} ({w_temp:.1f}°C, {w_precip:.1f} mm/h live rain, {w_wind:.1f} km/h wind, {w_hum}% humidity). Active precipitation is rapidly saturating regolith. "
        else:
            weather_snippet = f"Current condition: {w_desc} ({w_temp:.1f}°C, calm/moderate wind {w_wind:.1f} km/h, {w_hum}% humidity) with no active cloudburst. "

    if adjusted_score >= 80:
        advice_text = (
            f"URGENT: Severe landslide hazard detected (Score {adjusted_score}, {band['name']}). "
            f"{loc_prefix}{weather_snippet}"
            f"Steep mountain incline ({slope_val:.1f}°) combined with hydrometeorological saturation ({rain_3d_val:.1f} mm recent rain) "
            f"creates high probability of active debris flow or translational slope failure. "
            f"AI Companion Advice: Immediate avoidance recommended. Seek valley bypass or shelter away from slope toes until rain ceases."
        )
    elif adjusted_score >= 55:
        advice_text = (
            f"WARNING: High terrain vulnerability (Score {adjusted_score}, {band['name']}). "
            f"{loc_prefix}{weather_snippet}"
            f"Rainfall accumulation ({rain_3d_val:.1f} mm) is destabilizing mountain regolith on this {slope_val:.1f}° slope. "
            f"AI Companion Advice: For {profile_clean.title()} travelers, proceed with heightened vigilance or reroute along a flatter ridge corridor."
        )
    elif adjusted_score >= 25:
        advice_text = (
            f"ADVISORY: Moderate susceptibility (Score {adjusted_score}, {band['name']}). "
            f"{loc_prefix}{weather_snippet}"
            f"Terrain conditions are currently stable, but watch for localized roadside rockfalls if rain accelerates on the {slope_val:.1f}° gradient. "
            f"AI Companion Advice: Maintain normal precautions; monitor official meteorological bulletins."
        )
    else:
        advice_text = (
            f"NOTICE: Ground stability is favorable (Score {adjusted_score}, {band['name']}). "
            f"{loc_prefix}{weather_snippet}"
            f"Gentle terrain ({slope_val:.1f}°) and light rainfall ({rain_3d_val:.1f} mm) present low hazard risk. "
            f"AI Companion Advice: Ideal travel conditions along this section."
        )

    # 7. Formulate Output Payload
    output = {
        "score": adjusted_score,
        "base_score": base_score,
        "band": band["name"],
        "band_label": band["label"],
        "band_action": band["action"],
        "confidence": confidence,
        "profile": profile_clean,
        "profile_multiplier": multiplier,
        "terrain_zone": terrain_zone or {
            "name": "Moderate Slopes",
            "cluster_id": 1,
            "description": "Mid-elevation hilly terrain with moderate gradient.",
            "memberships": {"Low": 0.20, "Moderate": 0.65, "High": 0.15}
        },
        "top_factors": top_factors,
        "advice": advice_text,
        "disclaimer": "Model-derived relative estimates from public satellite/weather data. Never a guarantee of safety. Check GSI/IMD/NDMA warnings."
    }

    # Print summary as requested by Section 5.8
    logger.info(
        f"[AI Companion] Score: {adjusted_score} ({band['name']}) | Profile: {profile_clean} | "
        f"Confidence: {confidence} | Advice: {advice_text[:80]}..."
    )
    return output


if __name__ == "__main__":
    sample = {"slope_deg": 32.5, "rain_3d": 85.0, "elevation_m": 1850.0, "soil_wetness_top": 0.85}
    guide = ai_companion_guide(sample, profile="family")
    print(guide["advice"])
