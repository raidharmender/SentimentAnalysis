import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from typing import Dict, List
import os
import logging

from app.logging_config import get_dashboard_logger

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_DIR = "uploads"

# Initialize logger
logger = get_dashboard_logger()

def main():
    logger.info("Starting Streamlit dashboard")
    st.set_page_config(
        page_title="Sentiment Analysis Dashboard",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Audio Sentiment Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload & Analyze", "View Results", "Statistics", "System Status"]
    )
    
    logger.info(f"User navigated to page: {page}")
    
    if page == "Upload & Analyze":
        upload_and_analyze_page()
    elif page == "View Results":
        view_results_page()
    elif page == "Statistics":
        statistics_page()
    elif page == "System Status":
        system_status_page()

def upload_and_analyze_page():
    st.header("Upload & Analyze Audio")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'opus'],
        help="Supported formats: WAV, MP3, FLAC, M4A, OPUS"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Analysis options
        st.subheader("Analysis Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            save_processed = st.checkbox("Save processed audio", value=True)
        
        with col2:
            # Language selection
            language_options = {
                "Auto-detect (Improved)": "auto",
                "English": "en",
                "Welsh": "cy",
                "Irish": "ga",
                "Scottish Gaelic": "gd",
                "French": "fr",
                "German": "de",
                "Spanish": "es",
                "Italian": "it",
                "Portuguese": "pt",
                "Dutch": "nl",
                "Russian": "ru",
                "Chinese": "zh",
                "Japanese": "ja",
                "Korean": "ko"
            }
            
            selected_language = st.selectbox(
                "Language",
                options=list(language_options.keys()),
                index=0,  # Default to auto-detect
                help="Select the language of the audio. 'Auto-detect' uses improved detection for English accents."
            )
            
            language_code = language_options[selected_language]
        
        with col3:
            if st.button("Analyze Sentiment", type="primary"):
                analyze_audio(uploaded_file, save_processed, language_code)

def analyze_audio(uploaded_file, save_processed, language_code="auto"):
    """Analyze uploaded audio file"""
    logger.info(f"Starting audio analysis via dashboard for file: {uploaded_file.name}")
    logger.info(f"Analysis parameters: save_processed={save_processed}, language={language_code}")
    
    try:
        with st.spinner("Analyzing audio..."):
            # Prepare file for upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            params = {
                "save_processed_audio": save_processed,
                "language": language_code
            }
            
            # Make API request
            logger.info(f"Making API request to: {API_BASE_URL}/analyze")
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Audio analysis completed successfully for file: {uploaded_file.name}")
                logger.info(f"Analysis ID: {result.get('analysis_id')}, Sentiment: {result.get('sentiment', {}).get('overall_sentiment')}")
                display_analysis_results(result)
            else:
                logger.error(f"API analysis failed for file {uploaded_file.name}: {response.status_code} - {response.text}")
                st.error(f"Analysis failed: {response.text}")
                
    except Exception as e:
        logger.error(f"Error during dashboard audio analysis for file {uploaded_file.name}: {str(e)}", exc_info=True)
        st.error(f"Error during analysis: {str(e)}")

def display_analysis_results(result):
    """Display analysis results"""
    st.success("Analysis completed successfully!")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{result['duration']:.2f}s")
    with col2:
        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
    with col3:
        # Language info
        language_name = {
            "en": "English",
            "cy": "Welsh", 
            "ga": "Irish",
            "gd": "Scottish Gaelic",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ms": "Malay"
        }.get(result['transcription']['language'], result['transcription']['language'])
        
        st.metric(
            "Detected Language",
            language_name,
            delta=f"Confidence: {result['transcription']['confidence']:.1%}"
        )
    with col4:
        sentiment_color = {
            "positive": "green",
            "negative": "red",
            "neutral": "gray"
        }.get(result['sentiment']['overall_sentiment'], "gray")
        st.metric(
            "Overall Sentiment",
            result['sentiment']['overall_sentiment'].title(),
            delta=f"{result['sentiment']['score']:.3f}"
        )
    
    # Transcription
    st.subheader("📝 Transcription")
    st.text_area(
        "Transcribed Text",
        result['transcription']['text'],
        height=150,
        disabled=True
    )
    
    # Sentiment details
    st.subheader("😊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        # Sentiment score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['sentiment']['score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment Score"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence and details
        st.metric("Confidence", f"{result['sentiment']['confidence']:.2%}")
        
        # Show tool used
        tool = result['sentiment']['details'].get('tool') if isinstance(result['sentiment']['details'], dict) else None
        if tool:
            st.write(f"**Sentiment Tool Used:** `{tool}`")
        # Show detailed output for Mandarin, Malay, English
        details = result['sentiment']['details']
        if details:
            if tool == 'cnsenti+cntext':
                st.write("**CNSenti Sentiment:**", details.get('cnsenti_sentiment'))
                st.write("**CNSenti Emotion:**", details.get('cnsenti_emotion'))
                st.write("**CnText:**", details.get('cntext'))
            elif tool == 'malaya':
                st.write("**Malaya Sentiment:**", details.get('sentiment'))
            elif tool == 'VADER':
                st.write("**VADER Scores:**", details.get('sentiment'))
            else:
                # Fallback: show all details
                st.write("**Model Details:**")
                for k, v in details.items():
                    st.write(f"- {k}: {v}")
    
    # Segment analysis
    if result.get('segments'):
        st.subheader("📊 Segment Analysis")
        
        # Create segment dataframe
        segments_df = pd.DataFrame(result['segments'])
        segments_df['duration'] = segments_df['end'] - segments_df['start']
        
        # Segment sentiment over time
        fig = px.line(
            segments_df,
            x='start',
            y='score',
            title="Sentiment Score Over Time",
            labels={'start': 'Time (seconds)', 'score': 'Sentiment Score'}
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment table
        st.write("**Segment Details:**")
        display_df = segments_df[['start', 'end', 'text', 'sentiment', 'score']].copy()
        display_df['text'] = display_df['text'].str[:50] + "..."
        st.dataframe(display_df, use_container_width=True)

def view_results_page():
    st.header("View Analysis Results")
    
    try:
        # Get analyses
        response = requests.get(f"{API_BASE_URL}/analyses")
        if response.status_code == 200:
            data = response.json()
            analyses = data['analyses']
            
            if not analyses:
                st.info("No analyses found. Upload an audio file to get started!")
                return
            
            # Create dataframe
            df = pd.DataFrame(analyses)
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                sentiment_filter = st.selectbox(
                    "Filter by sentiment",
                    ["All"] + list(df['sentiment_label'].unique())
                )
            
            with col2:
                date_filter = st.date_input(
                    "Filter by date",
                    value=datetime.now().date()
                )
            
            # Apply filters
            if sentiment_filter != "All":
                df = df[df['sentiment_label'] == sentiment_filter]
            
            df = df[df['created_at'].dt.date == date_filter]
            
            # Display results
            st.write(f"Showing {len(df)} analyses")
            
            for _, analysis in df.iterrows():
                with st.expander(f"{analysis['filename']} - {analysis['sentiment_label'].title()}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Sentiment:** {analysis['sentiment_label'].title()}")
                    with col2:
                        st.write(f"**Score:** {analysis['sentiment_score']:.3f}")
                    with col3:
                        st.write(f"**Processing Time:** {analysis['processing_time']:.2f}s")
                    
                    if st.button(f"View Details", key=analysis['id']):
                        view_analysis_details(analysis['id'])
        else:
            st.error("Failed to fetch analyses")
            
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")

def view_analysis_details(analysis_id):
    """View detailed analysis results"""
    try:
        response = requests.get(f"{API_BASE_URL}/analyses/{analysis_id}")
        if response.status_code == 200:
            result = response.json()
            display_analysis_results(result)
        else:
            st.error("Failed to fetch analysis details")
    except Exception as e:
        st.error(f"Error loading analysis details: {str(e)}")

def statistics_page():
    st.header("📈 Sentiment Statistics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/statistics")
        if response.status_code == 200:
            stats = response.json()
            
            # Overall metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyses", stats['total_analyses'])
            with col2:
                st.metric("Average Score", f"{stats['average_sentiment_score']:.3f}")
            with col3:
                # Most common sentiment
                distribution = stats['sentiment_distribution']
                most_common = max(distribution, key=distribution.get)
                st.metric("Most Common", most_common.title())
            
            # Sentiment distribution pie chart
            if stats['total_analyses'] > 0:
                fig = px.pie(
                    values=list(distribution.values()),
                    names=list(distribution.keys()),
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution table
                st.write("**Detailed Distribution:**")
                dist_df = pd.DataFrame([
                    {"Sentiment": k.title(), "Count": v, "Percentage": f"{v/stats['total_analyses']*100:.1f}%"}
                    for k, v in distribution.items()
                ])
                st.dataframe(dist_df, use_container_width=True)
            else:
                st.info("No analyses available for statistics")
        else:
            st.error("Failed to fetch statistics")
            
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")

def system_status_page():
    st.header("🔧 System Status")
    
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        if response.status_code == 200:
            status = response.json()
            
            # Audio processor status
            st.subheader("Audio Processor")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Target Sample Rate:** {status['audio_processor']['target_sample_rate']} Hz")
            with col2:
                st.write(f"**Target Channels:** {status['audio_processor']['target_channels']}")
            
            # Transcription status
            st.subheader("Transcription Service")
            trans_status = status['transcription']
            st.write(f"**Model:** {trans_status['model_name']}")
            st.write(f"**Status:** {trans_status['status']}")
            if trans_status['status'] == 'loaded':
                st.write(f"**Device:** {trans_status['device']}")
            
            # Sentiment analysis status
            st.subheader("Sentiment Analysis")
            sent_status = status['sentiment_analysis']
            st.write(f"**Hugging Face Model:** {sent_status['huggingface_model']}")
            st.write(f"**Hugging Face Loaded:** {sent_status['huggingface_loaded']}")
            st.write(f"**VADER Available:** {sent_status['vader_available']}")
            st.write(f"**TextBlob Available:** {sent_status['textblob_available']}")
            st.write(f"**Device:** {sent_status['device']}")
            
            # Storage status
            st.subheader("Storage")
            storage = status['storage']
            st.write(f"**Upload Directory:** {storage['upload_dir']}")
            st.write(f"**Processed Directory:** {storage['processed_dir']}")
            
            # Health check
            health_response = requests.get(f"{API_BASE_URL}/health")
            if health_response.status_code == 200:
                st.success("✅ API is healthy")
            else:
                st.error("❌ API health check failed")
                
        else:
            st.error("Failed to fetch system status")
            
    except Exception as e:
        st.error(f"Error loading system status: {str(e)}")

if __name__ == "__main__":
    main() 