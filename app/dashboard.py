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
from app.template_utils import (
    load_css, format_sentiment_header, format_compact_sentiment_header,
    format_metrics_grid, format_section_header, get_sentiment_config
)

# Configuration
API_BASE_URL = "http://localhost:8000"
UPLOAD_DIR = "uploads"

# Initialize logger
logger = get_dashboard_logger()

def main():
    logger.info("Starting Streamlit dashboard")
    
    st.set_page_config(
        page_title="Artificial Intelligence Telecall Analyst",
        layout="wide"
    )
    
    # Load and inject CSS
    css = load_css()
    st.markdown(f"""
    <style>
    {css}
    </style>
    """, unsafe_allow_html=True)
    
    # Force left alignment container
    st.markdown("""
    <div style="text-align: left; width: 100%; max-width: 100%;">
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='dashboard-header'>Artificial Intelligence Telecall Analyst</h1>", unsafe_allow_html=True)
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
    
    # Close the left alignment container
    st.markdown("</div>", unsafe_allow_html=True)

def upload_and_analyze_page():
    st.markdown("<h1 class='page-header'>Upload & Analyze Audio</h1>", unsafe_allow_html=True)
    
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
        st.markdown("")       
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
    """Display analysis results using templates"""
    # Handle different result structures (from API vs from database)
    if 'sentiment' in result and isinstance(result['sentiment'], dict):
        # From API analysis
        overall_sentiment = result['sentiment']['overall_sentiment'].lower()
        sentiment_score = result['sentiment']['score']
        transcription_text = result['transcription']['text']
        transcription_language = result['transcription']['language']
        transcription_confidence = result['transcription']['confidence']
    else:
        # From database (detailed view)
        overall_sentiment = result['sentiment_label'].lower()
        sentiment_score = result['sentiment_score']
        transcription_text = result['transcript']
        transcription_language = 'en'  # Default, not stored in database
        transcription_confidence = result['transcription_confidence']
    
    # Display each section using templates
    display_sentiment_header(overall_sentiment, sentiment_score)
    display_analysis_metrics(result, transcription_language, transcription_confidence, overall_sentiment, sentiment_score)
    display_segment_analysis(result)

def display_sentiment_header(overall_sentiment, sentiment_score):
    """Display sentiment header using template"""
    sentiment_html = format_sentiment_header(overall_sentiment, sentiment_score)
    st.markdown(sentiment_html, unsafe_allow_html=True)

def display_analysis_metrics(result, transcription_language, transcription_confidence, overall_sentiment, sentiment_score):
    """Display analysis metrics using template"""
    # Language info for display
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
    }.get(transcription_language, transcription_language)
    
    # Display metrics header
    metrics_header = format_section_header("📊 Analysis Metrics")
    st.markdown(metrics_header, unsafe_allow_html=True)
    
    # Display metrics using Streamlit columns instead of HTML grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">⏱️ Duration</h3>
            <p class="metric-value">{result['duration']:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">⚡ Processing Time</h3>
            <p class="metric-value">{result['processing_time']:.2f}s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">🌐 Detected Language</h3>
            <p class="metric-value">{language_name}</p>
            <p class="metric-subtitle">Confidence: {transcription_confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sentiment_config = get_sentiment_config(overall_sentiment)
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">💭 Overall Sentiment</h3>
            <p class="metric-value {sentiment_config['class']}">{overall_sentiment.title()}</p>
            <p class="metric-subtitle">Score: {sentiment_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def display_segment_analysis(result):
    """Display segment analysis"""
    if result.get('segments'):
        # Segment analysis header
        segment_header = format_section_header("📊 Segment Analysis", large=True)
        st.markdown(segment_header, unsafe_allow_html=True)
        
        # Create segment dataframe - handle different segment structures
        segments = result['segments']
        if segments and len(segments) > 0:
            # Check if segments have 'start'/'end' (API format) or 'start_time'/'end_time' (DB format)
            if 'start' in segments[0]:
                # API format
                segments_df = pd.DataFrame(segments)
                segments_df['duration'] = segments_df['end'] - segments_df['start']
                time_col = 'start'
                text_col = 'text'
                sentiment_col = 'sentiment'
                score_col = 'score'
            else:
                # Database format
                segments_df = pd.DataFrame(segments)
                segments_df['duration'] = segments_df['end_time'] - segments_df['start_time']
                segments_df['start'] = segments_df['start_time']
                segments_df['end'] = segments_df['end_time']
                segments_df['text'] = segments_df['transcript']
                segments_df['sentiment'] = segments_df['sentiment_label']
                segments_df['score'] = segments_df['sentiment_score']
                time_col = 'start'
                text_col = 'text'
                sentiment_col = 'sentiment'
                score_col = 'score'
            
            # Segment sentiment over time
            fig = px.line(
                segments_df,
                x=time_col,
                y=score_col,
                title="Sentiment Score Over Time",
                labels={'start': 'Time (seconds)', 'score': 'Sentiment Score'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                height=500,
                title_font_size=24,
                xaxis_title_font_size=18,
                yaxis_title_font_size=18,
                font=dict(size=16)
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Segment table header
            segment_details_header = format_section_header("📋 Segment Details", large=True)
            st.markdown(segment_details_header, unsafe_allow_html=True)
            
            # Display segment table
            display_df = segments_df[['start', 'end', text_col, sentiment_col, score_col]].copy()
            display_df['text'] = display_df[text_col].str[:150] + "..."  # Show more text
            st.dataframe(display_df, use_container_width=True, height=500)

def view_results_page():
    st.markdown("<h1 class='page-header'>View Analysis Results</h1>", unsafe_allow_html=True)
    
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
                    ["All"] + list(df['sentiment'].unique())
                )
            
            with col2:
                date_filter = st.date_input(
                    "Filter by date",
                    value=datetime.now().date()
                )
            
            # Apply filters
            if sentiment_filter != "All":
                df = df[df['sentiment'] == sentiment_filter]
            
            df = df[df['created_at'].dt.date == date_filter]
            
            # Display results
            st.write(f"Showing {len(df)} analyses")
            
            for _, analysis in df.iterrows():
                # Create compact sentiment header for each analysis
                sentiment_label = analysis['sentiment'].lower()
                sentiment_score = 0.0  # Will be updated when viewing details
                
                # Create compact sentiment header using template
                compact_header_html = format_compact_sentiment_header(sentiment_label, sentiment_score)
                
                with st.expander(f"{analysis['filename']} - {analysis['sentiment'].title()}"):
                    st.markdown(compact_header_html, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Sentiment:** {analysis['sentiment'].title()}")
                    with col2:
                        st.write(f"**Duration:** {analysis['duration']:.2f}s")
                    with col3:
                        st.write(f"**Processing Time:** {analysis['processing_time']:.2f}s")
                    
                    if st.button(f"View Details", key=analysis['analysis_id']):
                        view_analysis_details(analysis['analysis_id'])
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
            # Add a separator and title for the detailed view
            st.markdown("---")
            st.subheader("📊 Detailed Analysis Results")
            display_analysis_results(result)
        else:
            st.error("Failed to fetch analysis details")
    except Exception as e:
        st.error(f"Error loading analysis details: {str(e)}")

def statistics_page():
    st.markdown("<h1 class='page-header'>📈 Call Analysis Statistics</h1>", unsafe_allow_html=True)
    
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
    st.markdown("<h1 class='page-header'>🔧 System Status</h1>", unsafe_allow_html=True)
    
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
            st.write(f"**Status:** {'Loaded' if trans_status['model_loaded'] else 'Not Loaded'}")
            st.write(f"**Device:** {trans_status['device']}")
            st.write(f"**CUDA Available:** {trans_status['cuda_available']}")
            st.write(f"**Supported Languages:** {trans_status['supported_languages']}")
            
            # Sentiment analysis status
            st.subheader("Sentiment Analysis")
            sent_status = status['sentiment_analysis']
            st.write(f"**Multi-Tool Analyzer:** {sent_status.get('multi_tool_analyzer', False)}")
            st.write(f"**VADER Available:** {sent_status.get('vader_available', False)}")
            st.write(f"**SnowNLP Available:** {sent_status.get('snownlp_available', False)}")
            st.write(f"**CNText Available:** {sent_status.get('cntext_available', False)}")
            st.write(f"**Malaya Available:** {sent_status.get('malaya_available', False)}")
            st.write(f"**Supported Languages:** {', '.join(sent_status.get('supported_languages', []))}")
            
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