"""
Permafrost stability analysis service
"""
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from datetime import datetime
from ..models.permafrost import PermafrostSite
from geoalchemy2.elements import WKTElement
import numpy as np


def analyze_permafrost_stability(
    db: Session,
    location: Dict[str, float],
    site_type: str,
    site_name: str,
    building_specs: Dict[str, Any] = None,
    user_id: str = None
) -> Dict[str, Any]:
    """
    Analyze permafrost stability for a site

    Args:
        location: {"lat": 70.0, "lon": -120.0}
        site_type: building, mine, infrastructure
        site_name: Site identifier
        building_specs: Optional building specifications
        user_id: User ID

    Returns:
        Stability analysis results
    """
    # Simplified analysis - in production, use satellite data, temperature models, etc.

    # Calculate stability score based on latitude and historical data
    stability_score = _calculate_stability_score(location, site_type)

    # Estimate temperature
    temperature_c = _estimate_permafrost_temperature(location)

    # Determine alert level
    alert_level = _determine_alert_level(stability_score, temperature_c)

    # Generate recommendations
    recommendations = _generate_recommendations(stability_score, alert_level, site_type)

    # Save to database
    site = PermafrostSite(
        site_name=site_name,
        site_type=site_type,
        location=WKTElement(f"POINT({location['lon']} {location['lat']})", srid=4326),
        stability_score=stability_score,
        temperature_c=temperature_c,
        last_analysis=datetime.utcnow(),
        alert_level=alert_level,
        user_id=user_id
    )

    db.add(site)
    db.commit()
    db.refresh(site)

    return {
        "site_id": site.id,
        "stability_score": stability_score,
        "temperature_c": temperature_c,
        "alert_level": alert_level,
        "recommendations": recommendations,
        "last_analysis": site.last_analysis.isoformat()
    }


def get_user_sites(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """Get all monitored sites for a user"""
    sites = db.query(PermafrostSite).filter(
        PermafrostSite.user_id == user_id
    ).all()

    return [
        {
            "site_id": site.id,
            "site_name": site.site_name,
            "site_type": site.site_type,
            "stability_score": site.stability_score,
            "alert_level": site.alert_level,
            "last_analysis": site.last_analysis.isoformat()
        }
        for site in sites
    ]


def get_alerts(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """Get active stability alerts for user"""
    sites = db.query(PermafrostSite).filter(
        PermafrostSite.user_id == user_id,
        PermafrostSite.alert_level.in_(["warning", "critical"])
    ).all()

    return [
        {
            "site_id": site.id,
            "site_name": site.site_name,
            "alert_level": site.alert_level,
            "stability_score": site.stability_score,
            "temperature_c": site.temperature_c
        }
        for site in sites
    ]


def _calculate_stability_score(location: Dict[str, float], site_type: str) -> float:
    """Calculate permafrost stability score"""
    # Simplified model
    # Higher latitude generally = more stable permafrost
    # But warming trends reduce stability
    base_score = min(100, max(0, (location["lat"] - 60) * 2))

    # Adjust for site type
    if site_type == "mine":
        base_score -= 15  # Mining activity reduces stability
    elif site_type == "building":
        base_score -= 5

    # Add some variation
    variation = np.random.rand() * 10 - 5
    return max(0, min(100, base_score + variation))


def _estimate_permafrost_temperature(location: Dict[str, float]) -> float:
    """Estimate permafrost temperature"""
    # Simplified model
    # Temperature decreases with latitude
    base_temp = 5 - (location["lat"] - 60) * 0.5
    return base_temp + np.random.rand() * 2 - 1


def _determine_alert_level(stability_score: float, temperature_c: float) -> str:
    """Determine alert level"""
    if stability_score < 40 or temperature_c > -0.5:
        return "critical"
    elif stability_score < 60 or temperature_c > -2.0:
        return "warning"
    else:
        return "stable"


def _generate_recommendations(
    stability_score: float,
    alert_level: str,
    site_type: str
) -> List[str]:
    """Generate recommendations based on analysis"""
    recommendations = []

    if alert_level == "critical":
        recommendations.append("Immediate structural assessment required")
        recommendations.append("Consider foundation reinforcement")
        recommendations.append("Install active cooling system")

    elif alert_level == "warning":
        recommendations.append("Increase monitoring frequency")
        recommendations.append("Review foundation design")
        recommendations.append("Plan for seasonal variations")

    else:
        recommendations.append("Continue regular monitoring")
        recommendations.append("Maintain current protocols")

    if site_type == "building":
        recommendations.append("Ensure proper insulation to prevent heat transfer")

    return recommendations
