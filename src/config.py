# Configuration for GreenSumm project

# Model Names
BASELINE_MODEL = "facebook/bart-large-cnn"
OPTIMIZED_MODEL = "sshleifer/distilbart-cnn-12-6"
FALLBACK_MODEL = "t5-small"

# Hardware Power Profiles (Watts)
POWER_PROFILES = {
    "Mobile CPU Approximation": 3.0,
    "Laptop CPU Approximation": 15.0,
    "GPU Approximation": 50.0
}

# Carbon Intensity (gCO2/kWh)
DEFAULT_CARBON_INTENSITY = 475.0

# Summary Length Defaults
DEFAULT_MIN_LENGTH = 30
DEFAULT_MAX_LENGTH = 150

# ROUGE Metrics
ROUGE_METRICS = ['rouge1', 'rouge2', 'rougeL']

# Project Info
PROJECT_TITLE = "GreenSumm: Sustainable Text Summarization on Mobile Devices"
TEAM_MEMBERS = "BE Final Year Project Team"
