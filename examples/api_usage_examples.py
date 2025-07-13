#!/usr/bin/env python3
"""
Comprehensive API Usage Examples for Sentiment Analysis System

This script demonstrates how to use all endpoints of the Sentiment Analysis API.
Make sure the API server is running on http://localhost:8000 before running these examples.
"""

import requests
import json
import time
import os
from typing import Dict, Any, Optional


class SentimentAnalysisAPI:
    """Client class for interacting with the Sentiment Analysis API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages."""
        response = self.session.get(f"{self.base_url}/languages")
        response.raise_for_status()
        return response.json()
    
    def analyze_audio_file(self, file_path: str, language: Optional[str] = None, 
                          save_processed: bool = True) -> Dict[str, Any]:
        """Analyze sentiment from audio file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'save_processed_audio': save_processed
            }
            if language:
                data['language'] = language
            
            response = self.session.post(f"{self.base_url}/analyze", files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def analyze_text(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Analyze sentiment from text."""
        data = {
            'text': text,
            'language': language
        }
        response = self.session.post(f"{self.base_url}/analyze/text", data=data)
        response.raise_for_status()
        return response.json()
    
    def get_all_analyses(self, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get list of all analyses with pagination."""
        params = {'limit': limit, 'offset': offset}
        response = self.session.get(f"{self.base_url}/analyses", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_analysis_by_id(self, analysis_id: int) -> Dict[str, Any]:
        """Get specific analysis by ID."""
        response = self.session.get(f"{self.base_url}/analyses/{analysis_id}")
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/statistics")
        response.raise_for_status()
        return response.json()
    
    def delete_analysis(self, analysis_id: int) -> Dict[str, Any]:
        """Delete specific analysis by ID."""
        response = self.session.delete(f"{self.base_url}/analyses/{analysis_id}")
        response.raise_for_status()
        return response.json()
    
    def graceful_shutdown(self) -> Dict[str, Any]:
        """Gracefully shutdown the API server."""
        response = self.session.post(f"{self.base_url}/shutdown")
        response.raise_for_status()
        return response.json()


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_json(data: Dict[str, Any], title: str = "Response"):
    """Print JSON data in a formatted way."""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2))


def main():
    """Run all API usage examples."""
    api = SentimentAnalysisAPI()
    
    try:
        # 1. Health Check
        print_section("1. Health Check")
        health = api.health_check()
        print_json(health, "Health Status")
        
        # 2. System Status
        print_section("2. System Status")
        status = api.get_system_status()
        print_json(status, "System Status")
        
        # 3. Supported Languages
        print_section("3. Supported Languages")
        languages = api.get_supported_languages()
        print_json(languages, "Supported Languages")
        
        # 4. Text Analysis Examples
        print_section("4. Text Analysis Examples")
        
        # English text analysis
        english_text = "I am absolutely delighted with the service! The staff was wonderful and everything exceeded my expectations."
        english_result = api.analyze_text(english_text, "en")
        print_json(english_result, "English Text Analysis")
        
        # Chinese text analysis (if supported)
        try:
            chinese_text = "这个产品非常好用，我很满意！"
            chinese_result = api.analyze_text(chinese_text, "zh")
            print_json(chinese_result, "Chinese Text Analysis")
        except Exception as e:
            print(f"Chinese analysis not available: {e}")
        
        # 5. Audio Analysis (if sample file exists)
        print_section("5. Audio Analysis")
        sample_audio = "incoming/Shoppee_Happy_1.opus"
        if os.path.exists(sample_audio):
            try:
                audio_result = api.analyze_audio_file(sample_audio, language="en")
                print_json(audio_result, "Audio Analysis Result")
            except Exception as e:
                print(f"Audio analysis failed: {e}")
        else:
            print(f"Sample audio file not found: {sample_audio}")
            print("Skipping audio analysis example")
        
        # 6. Get All Analyses
        print_section("6. Get All Analyses")
        analyses = api.get_all_analyses(limit=5)
        print_json(analyses, "Recent Analyses")
        
        # 7. Get Statistics
        print_section("7. System Statistics")
        stats = api.get_statistics()
        print_json(stats, "System Statistics")
        
        # 8. Get Specific Analysis (if available)
        if analyses.get('analyses'):
            print_section("8. Get Specific Analysis")
            first_analysis_id = analyses['analyses'][0]['analysis_id']
            specific_analysis = api.get_analysis_by_id(first_analysis_id)
            print_json(specific_analysis, f"Analysis ID {first_analysis_id}")
        
        # 9. Demonstrate different language options
        print_section("9. Language-Specific Analysis")
        
        # Neutral text
        neutral_text = "The product arrived on time and was as described."
        neutral_result = api.analyze_text(neutral_text, "en")
        print_json(neutral_result, "Neutral Text Analysis")
        
        # Negative text
        negative_text = "I'm very disappointed with the quality. The service was terrible and the product broke immediately."
        negative_result = api.analyze_text(negative_text, "en")
        print_json(negative_result, "Negative Text Analysis")
        
        print_section("Examples Completed Successfully!")
        print("\nNote: To test graceful shutdown, uncomment the following line:")
        print("# shutdown_result = api.graceful_shutdown()")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"Error during API testing: {e}")


if __name__ == "__main__":
    main() 