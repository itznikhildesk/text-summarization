from typing import Dict

def estimate_energy_joules(inference_time_seconds: float, power_watts: float) -> float:
    """Estimates energy consumption in Joules."""
    return power_watts * inference_time_seconds

def convert_joules_to_kwh(joules: float) -> float:
    """Converts Joules to kilowatt-hours (kWh)."""
    return joules / 3_600_000

def estimate_carbon_gco2(energy_kwh: float, carbon_intensity_g_per_kwh: float) -> float:
    """Estimates carbon footprint in grams of CO2."""
    return energy_kwh * carbon_intensity_g_per_kwh

def calculate_percentage_improvement(baseline: float, optimized: float) -> float:
    """Calculates the percentage improvement from baseline to optimized."""
    if baseline == 0:
        return 0.0
    return ((baseline - optimized) / baseline) * 100