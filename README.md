# GreenSumm: Sustainable Text Summarization on Mobile Devices

## Project Overview
GreenSumm is a research-oriented prototype demonstrating resource-efficient text summarization suitable for mobile and edge devices. It compares baseline transformer models with optimized/lightweight versions, measuring efficiency metrics aligned with Green AI principles.

## Features
- **Text & PDF Input**: Paste raw text or upload PDF documents.
- **Model Comparison**: Compare baseline (BART-Large) vs. optimized (DistilBART/Quantized) models.
- **Resource Tracking**: Measure inference time, peak memory usage, and estimated model size.
- **Sustainability Metrics**: Estimate energy consumption (Joules) and carbon footprint (gCO2).
- **Quality Evaluation**: Calculate ROUGE scores when a reference summary is provided.
- **Visual Analytics**: Interactive Plotly charts for metric comparison and trade-off analysis.
- **Smart Recommendations**: Automatic recommendation based on efficiency gains.

## Tech Stack
- **UI**: Streamlit
- **ML Framework**: Hugging Face Transformers, PyTorch
- **Data Handling**: Pandas, NumPy
- **Visualizations**: Plotly
- **System Metrics**: psutil
- **PDF Extraction**: PyMuPDF (fitz)

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd greensumm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Explanation of Metrics
- **Inference Time**: Time taken by the model to generate the summary.
- **Memory Usage**: Difference in system RAM usage before and after inference.
- **Model Size**: Estimated size of the model parameters in memory.
- **Energy Consumption**: Estimated as `Power (Watts) * Time (Seconds)`.
- **Carbon Footprint**: Estimated using local carbon intensity metrics.
- **ROUGE Score**: Measures overlap between machine-generated and human-reference summaries.
- **Green Efficiency Score**: A custom metric combining speed, memory, and energy efficiency.

## Limitations
- Energy and carbon values are estimates based on hardware profiles, not direct hardware measurements.
- Real mobile deployment would require specialized formats like TFLite or ONNX.
- Accuracy/Quality depends heavily on the chosen model and text domain.

## Future Scope
- Integration with Android/iOS via PyTorch Mobile.
- Support for more languages.
- Real-time battery impact monitoring.
- Offline-first capabilities for privacy and sustainability.
