"""Feature-to-Plain-Language Translation Module.

Translates technical hydrometeorological and topographic features into clear,
empathic, plain-language sentences with directional impacts and provenance tags (ADR 0009).
"""
from typing import Dict, Any, List


FEATURE_PLAIN_NAMES = {
    "slope_deg": "Terrain Slope",
    "elevation_m": "Ground Elevation",
    "roughness": "Terrain Ruggedness",
    "relief_m": "Local Topographic Relief",
    "curvature": "Slope Curvature",
    "aspect_sin": "East-West Slope Aspect",
    "aspect_cos": "North-South Slope Aspect",
    "rain_1d": "24h Recent Rainfall",
    "rain_3d": "3-Day Rainfall Accumulation",
    "rain_7d": "7-Day Cumulative Rain",
    "rain_14d": "14-Day Cumulative Rain",
    "rain_30d": "30-Day Monthly Rainfall",
    "rain_max_day_30d": "Peak Single-Day Storm Rain",
    "api_30d": "Antecedent Precipitation Index (Moisture Memory)",
    "soil_wetness_top": "Surface Soil Water Saturation",
    "soil_wetness_root_7d": "Root-Zone Soil Saturation",
    "distance_to_water_m": "Distance to Water Channel",
    "log_dist_to_water": "Log Distance to Water",
    "month_sin": "Seasonal Monsoon Cycle",
    "month_cos": "Annual Temperature / Dry Cycle",
    "region_group": "Geographic Region Zone"
}


def explain_feature_impact(feat_name: str, value: float, shap_val: float = None, provenance: str = "observed") -> Dict[str, Any]:
    """Generate human-readable summary of a single feature's physical impact on hazard risk."""
    name_display = FEATURE_PLAIN_NAMES.get(feat_name, feat_name.replace("_", " ").title())
    is_risk_increasing = shap_val > 0 if shap_val is not None else False

    desc = ""
    # Specific physical plain-language translations
    if feat_name == "slope_deg":
        if value >= 25.0:
            desc = f"Steep mountain slope ({value:.1f}°), significantly increasing gravitational shear stress."
        elif value >= 15.0:
            desc = f"Moderate incline ({value:.1f}°), susceptible under heavy rain."
        else:
            desc = f"Gentle or flat terrain ({value:.1f}°), lowering slide potential."

    elif feat_name == "rain_3d":
        if value >= 50.0:
            desc = f"Heavy 3-day rainfall accumulation ({value:.1f} mm), elevating pore-water pressure."
        elif value >= 20.0:
            desc = f"Moderate 3-day rain ({value:.1f} mm), dampening surface colluvium."
        else:
            desc = f"Low recent rainfall ({value:.1f} mm in 3 days)."

    elif feat_name == "rain_1d":
        if value >= 30.0:
            desc = f"Intense 24h downpour ({value:.1f} mm) triggering rapid runoff."
        else:
            desc = f"24h rainfall: {value:.1f} mm."

    elif feat_name == "rain_30d":
        if value >= 250.0:
            desc = f"Heavy cumulative monsoon rain ({value:.1f} mm over 30 days) saturating the regolith."
        else:
            desc = f"30-day cumulative precipitation: {value:.1f} mm."

    elif feat_name == "api_30d":
        if value >= 40.0:
            desc = f"Elevated antecedent precipitation index ({value:.1f}), indicating long ground memory of past storms."
        else:
            desc = f"Normal antecedent moisture memory index ({value:.1f})."

    elif feat_name in ["soil_wetness_top", "soil_wetness_root_7d"]:
        pct = value * 100.0
        if pct >= 75.0:
            desc = f"Near-saturated ground moisture ({pct:.0f}%), reducing soil cohesion and shear resistance."
        elif pct >= 50.0:
            desc = f"Moderate ground moisture ({pct:.0f}% saturation)."
        else:
            desc = f"Relatively dry soil ({pct:.0f}% saturation)."

    elif feat_name == "roughness":
        if value >= 25.0:
            desc = f"Highly rugged and fractured terrain (ruggedness {value:.1f}m)."
        else:
            desc = f"Smooth or rolling terrain profile (ruggedness {value:.1f}m)."

    elif feat_name == "relief_m":
        if value >= 200.0:
            desc = f"Steep canyon or valley relief ({value:.0f} m elevation change in 500m)."
        else:
            desc = f"Low local relief ({value:.0f} m)."

    elif feat_name == "distance_to_water_m":
        if value <= 200.0:
            desc = f"Immediate proximity to major river channel ({value:.0f}m), prone to toe-cutting erosion."
        elif value <= 1000.0:
            desc = f"Within {value:.0f}m of major river or lake shoreline."
        else:
            desc = f"Far from major waterways ({value/1000.0:.1f} km)."

    elif feat_name == "elevation_m":
        desc = f"Elevation of {value:.0f} meters above sea level."

    else:
        desc = f"{name_display}: {value} (impact: {'elevates risk' if is_risk_increasing else 'mitigates risk'})."

    return {
        "feature": feat_name,
        "label": name_display,
        "value": value,
        "shap_value": round(shap_val, 4) if shap_val is not None else 0.0,
        "direction": "up" if is_risk_increasing else "down",
        "description": desc,
        "provenance": provenance
    }
