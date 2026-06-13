import streamlit as st

def render_header():
    """Renders the professional title header."""
    st.title("🌱 GreenSumm")
    st.subheader("Sustainable Text Summarization for Mobile & Edge Devices")
    st.markdown("---")

def render_metric_card(label: str, value: str, delta: str = None, help_text: str = None):
    """Renders a metric card."""
    st.metric(label=label, value=value, delta=delta, help=help_text)

def render_recommendation_box(recommendation: str):
    """Renders the recommendation box."""
    st.success("### 💡 AI Recommendation")
    st.write(recommendation)

def render_footer():
    """Renders the footer."""
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey;'>"
        "GreenSumm | BE Final Year Project | Built with Streamlit & Hugging Face"
        "</div>",
        unsafe_allow_html=True
    )
