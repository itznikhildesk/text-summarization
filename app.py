import streamlit as st
import pandas as pd
import time
import os
from src.config import (
    BASELINE_MODEL, OPTIMIZED_MODEL, POWER_PROFILES, 
    DEFAULT_CARBON_INTENSITY, PROJECT_TITLE, DEFAULT_MAX_LENGTH, DEFAULT_MIN_LENGTH
)
from src.pdf_utils import extract_text_from_pdf
from src.text_utils import clean_text, count_words, calculate_compression_ratio
from src.summarizer import get_summarizer
from src.metrics import (
    calculate_rouge, get_memory_usage, calculate_model_size, calculate_efficiency_score
)
from src.energy import (
    estimate_energy_joules, convert_joules_to_kwh, 
    estimate_carbon_gco2, calculate_percentage_improvement
)
from src.charts import (
    create_comparison_bar_chart, create_rouge_comparison_chart, create_tradeoff_scatter_plot
)
from src.ui_components import (
    render_header, render_metric_card, render_recommendation_box, render_footer
)

# Page Configuration
st.set_page_config(
    page_title="GreenSumm",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'baseline_done' not in st.session_state:
    st.session_state.baseline_done = False
if 'optimized_done' not in st.session_state:
    st.session_state.optimized_done = False

def run_summarization(model_type, text, min_len, max_len, use_quant, power_val, carbon_intensity, ref_summary=None):
    model_name = BASELINE_MODEL if model_type == "Baseline" else OPTIMIZED_MODEL
    
    with st.spinner(f"Running {model_type} model..."):
        # Measure starting memory
        mem_before = get_memory_usage()
        start_time = time.perf_counter()
        
        # Load model & summarize
        summarizer = get_summarizer(model_name, use_quantization=use_quant if model_type == "Optimized" else False)
        summary = summarizer.summarize(text, min_length=min_len, max_length=max_len)
        
        # Measure ending metrics
        end_time = time.perf_counter()
        mem_after = get_memory_usage()
        
        # Calculations
        duration = end_time - start_time
        mem_used = max(0, mem_after - mem_before)
        model_size = calculate_model_size(summarizer.model, is_quantized=use_quant if model_type == "Optimized" else False)
        
        energy_j = estimate_energy_joules(duration, power_val)
        energy_kwh = convert_joules_to_kwh(energy_j)
        carbon_g = estimate_carbon_gco2(energy_kwh, carbon_intensity)
        
        comp_ratio = calculate_compression_ratio(text, summary)
        
        rouge_scores = None
        if ref_summary:
            rouge_scores = calculate_rouge(ref_summary, summary)
            
        efficiency = calculate_efficiency_score(duration, mem_used, energy_j)
        
        return {
            "summary": summary,
            "time": duration,
            "memory": mem_used,
            "model_size": model_size,
            "energy": energy_j,
            "carbon": carbon_g,
            "compression": comp_ratio,
            "rouge": rouge_scores,
            "efficiency": efficiency,
            "word_count": count_words(summary)
        }

# Sidebar
st.sidebar.title("🌱 GreenSumm Settings")
st.sidebar.markdown("---")

st.sidebar.subheader("Model Configuration")
power_mode = st.sidebar.selectbox("Hardware Profile", list(POWER_PROFILES.keys()), index=1)
power_value = POWER_PROFILES[power_mode]

carbon_intensity = st.sidebar.slider("Carbon Intensity (gCO2/kWh)", 0, 1000, int(DEFAULT_CARBON_INTENSITY))

st.sidebar.markdown("---")
st.sidebar.subheader("Summary Parameters")
min_length = st.sidebar.number_input("Min Length", 10, 100, DEFAULT_MIN_LENGTH)
max_length = st.sidebar.number_input("Max Length", 50, 500, DEFAULT_MAX_LENGTH)

st.sidebar.markdown("---")
if st.sidebar.button("Clear Results"):
    st.session_state.results = {}
    st.session_state.baseline_done = False
    st.session_state.optimized_done = False
    st.rerun()

st.sidebar.info(
    "**Green AI Tip**: Optimized models use less energy and produce fewer carbon emissions while maintaining similar utility."
)

# Main App
render_header()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Input", "📝 Summary", "📊 Metrics Dashboard", "🔄 Model Comparison", "ℹ️ About Project"
])

with tab1:
    st.header("Input Text for Summarization")
    
    input_method = st.radio("Choose Input Method", ["Paste Text", "Upload PDF"])
    
    input_text = ""
    if input_method == "Paste Text":
        input_text = st.text_area("Paste your text here:", height=300, placeholder="Enter text to summarize...")
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            extracted_text = extract_text_from_pdf(uploaded_file)
            if extracted_text:
                st.success("PDF Text Extracted Successfully!")
                input_text = st.text_area("Extracted Text Preview:", value=extracted_text, height=300)
            else:
                st.error("Could not extract text from the PDF. It might be an image-only PDF or corrupted.")

    if input_text:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Word Count:** {count_words(input_text)}")
        with col2:
            st.info(f"**Character Count:** {len(input_text)}")
            
        if count_words(input_text) < 50:
            st.warning("The text is quite short. Summarization might not be very effective.")

    st.markdown("---")
    st.subheader("Reference Summary (Optional)")
    ref_summary = st.text_area("Paste human/reference summary for ROUGE evaluation:", height=100)

with tab2:
    if not input_text:
        st.warning("Please provide input text in the 'Input' tab first.")
    else:
        st.header("Generate Summaries")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🚀 Run Baseline Model (BART-Large)", use_container_width=True):
                st.session_state.results["Baseline"] = run_summarization(
                    "Baseline", input_text, min_length, max_length, False, power_value, carbon_intensity, ref_summary
                )
                st.session_state.baseline_done = True
        
        with col_b:
            if st.button("🍃 Run Optimized Model (DistilBART + Quant)", use_container_width=True):
                st.session_state.results["Optimized"] = run_summarization(
                    "Optimized", input_text, min_length, max_length, True, power_value, carbon_intensity, ref_summary
                )
                st.session_state.optimized_done = True
        
        st.markdown("---")
        
        res_col1, res_col2 = st.columns(2)
        
        if st.session_state.baseline_done:
            with res_col1:
                st.subheader("Baseline Summary")
                st.write(st.session_state.results["Baseline"]["summary"])
                st.caption(f"Words: {st.session_state.results['Baseline']['word_count']} | Time: {st.session_state.results['Baseline']['time']:.2f}s")
        
        if st.session_state.optimized_done:
            with res_col2:
                st.subheader("Optimized Summary")
                st.write(st.session_state.results["Optimized"]["summary"])
                st.caption(f"Words: {st.session_state.results['Optimized']['word_count']} | Time: {st.session_state.results['Optimized']['time']:.2f}s")

with tab3:
    if not st.session_state.results:
        st.info("Run a model in the 'Summary' tab to see metrics.")
    else:
        st.header("In-depth Metrics")
        
        for model_name, res in st.session_state.results.items():
            with st.expander(f"Detailed Metrics for {model_name} Model", expanded=True):
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                with m_col1:
                    render_metric_card("Inference Time", f"{res['time']:.3f} s")
                    render_metric_card("Peak Memory", f"{res['memory']:.2f} MB")
                with m_col2:
                    render_metric_card("Energy Used", f"{res['energy']:.4f} J")
                    render_metric_card("CO2 Emission", f"{res['carbon']:.6f} g")
                with m_col3:
                    render_metric_card("Model Size", f"{res['model_size']:.1f} MB")
                    render_metric_card("Comp. Ratio", f"{res['compression']:.2%}")
                with m_col4:
                    render_metric_card("Efficiency Score", f"{res['efficiency']}/100")
                    if res['rouge']:
                        render_metric_card("ROUGE-L", f"{res['rouge']['rougeL']:.3f}")
                    else:
                        st.write("ROUGE: N/A")

with tab4:
    if not (st.session_state.baseline_done and st.session_state.optimized_done):
        st.info("Run BOTH models to see the comparison dashboard.")
    else:
        st.header("Performance Comparison")
        
        b = st.session_state.results["Baseline"]
        o = st.session_state.results["Optimized"]
        
        # Data preparation
        comparison_data = [
            {"Model": "Baseline", "Time (s)": b["time"], "Memory (MB)": b["memory"], "Energy (J)": b["energy"], "Carbon (g)": b["carbon"], "Size (MB)": b["model_size"], "Efficiency": b["efficiency"], "Quality": b["rouge"]["rougeL"] if b["rouge"] else 0.8},
            {"Model": "Optimized", "Time (s)": o["time"], "Memory (MB)": o["memory"], "Energy (J)": o["energy"], "Carbon (g)": o["carbon"], "Size (MB)": o["model_size"], "Efficiency": o["efficiency"], "Quality": o["rouge"]["rougeL"] if o["rouge"] else 0.75}
        ]
        df_comp = pd.DataFrame(comparison_data)
        
        # Graphs
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(create_comparison_bar_chart(df_comp, "Time (s)", "Latency Comparison", "Seconds"), use_container_width=True)
            st.plotly_chart(create_comparison_bar_chart(df_comp, "Energy (J)", "Energy Consumption", "Joules"), use_container_width=True)
        
        with g2:
            st.plotly_chart(create_comparison_bar_chart(df_comp, "Memory (MB)", "RAM Usage", "MB"), use_container_width=True)
            st.plotly_chart(create_comparison_bar_chart(df_comp, "Size (MB)", "Model Size in RAM", "MB"), use_container_width=True)
            
        st.markdown("---")
        
        # ROUGE Chart if available
        if b["rouge"] and o["rouge"]:
            st.subheader("Quality Metrics (ROUGE)")
            rouge_data = []
            for m_name in ["Baseline", "Optimized"]:
                for r_metric in ["rouge1", "rouge2", "rougeL"]:
                    rouge_data.append({
                        "Model": m_name,
                        "Metric": r_metric.upper(),
                        "Score": st.session_state.results[m_name]["rouge"][r_metric]
                    })
            df_rouge = pd.DataFrame(rouge_data)
            st.plotly_chart(create_rouge_comparison_chart(df_rouge), use_container_width=True)
        
        st.markdown("---")
        st.plotly_chart(create_tradeoff_scatter_plot(comparison_data), use_container_width=True)
        
        # Recommendation
        st.markdown("---")
        time_imp = calculate_percentage_improvement(b["time"], o["time"])
        energy_imp = calculate_percentage_improvement(b["energy"], o["energy"])
        mem_imp = calculate_percentage_improvement(b["memory"], o["memory"])
        
        rec_text = (
            f"The **Optimized model** is highly recommended for mobile deployment. "
            f"It achieved a **{time_imp:.1f}% reduction in latency**, "
            f"a **{energy_imp:.1f}% reduction in energy consumption**, and "
            f"uses **{mem_imp:.1f}% less peak memory** compared to the baseline model."
        )
        if b["rouge"] and o["rouge"]:
            q_diff = b["rouge"]["rougeL"] - o["rouge"]["rougeL"]
            rec_text += f" The quality trade-off (ROUGE-L) is only {q_diff:.3f}, which is acceptable for most mobile applications."
            
        render_recommendation_box(rec_text)

with tab5:
    st.header("About GreenSumm")
    st.write("""
    This project demonstrates sustainable text summarization for mobile and edge devices. 
    Instead of only focusing on summary accuracy, it also evaluates latency, memory usage, 
    model size, estimated energy consumption, and estimated carbon footprint. 
    This aligns with Green AI principles and shows the trade-off between summary quality and resource efficiency.
    """)
    
    st.subheader("Key Concepts")
    st.markdown("""
    - **Knowledge Distillation**: Training a smaller 'student' model to mimic a larger 'teacher' model.
    - **Quantization**: Reducing the precision of model weights (e.g., from FP32 to INT8) to save memory and speed up inference.
    - **Green AI**: A movement in AI research that prioritizes energy efficiency and environmental sustainability alongside accuracy.
    """)
    
    st.subheader("Limitations & Future Scope")
    col_l, col_f = st.columns(2)
    with col_l:
        st.info("**Limitations**")
        st.write("""
        - Energy and carbon are estimates, not direct measurements.
        - Requires specialized formats (ONNX/TFLite) for real mobile hardware.
        - Performance depends on the host system.
        """)
    with col_f:
        st.success("**Future Scope**")
        st.write("""
        - Integration with Android/iOS.
        - Real-time battery monitoring.
        - Multi-lingual support.
        - Offline summarization.
        """)

render_footer()
