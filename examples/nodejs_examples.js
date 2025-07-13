#!/usr/bin/env node

/**
 * Sentiment Analysis API - Node.js Examples
 * 
 * This script demonstrates how to use all endpoints of the Sentiment Analysis API
 * using Node.js and the 'axios' library.
 * 
 * Prerequisites:
 * - Node.js installed
 * - Run: npm install axios
 * - API server running on http://localhost:8000
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

class SentimentAnalysisAPI {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
        this.client = axios.create({
            baseURL,
            timeout: 30000, // 30 seconds
        });
    }

    async healthCheck() {
        const response = await this.client.get('/health');
        return response.data;
    }

    async getSystemStatus() {
        const response = await this.client.get('/status');
        return response.data;
    }

    async getSupportedLanguages() {
        const response = await this.client.get('/languages');
        return response.data;
    }

    async analyzeAudioFile(filePath, language = null, saveProcessed = true) {
        if (!fs.existsSync(filePath)) {
            throw new Error(`Audio file not found: ${filePath}`);
        }

        const formData = new FormData();
        formData.append('file', fs.createReadStream(filePath));
        formData.append('save_processed_audio', saveProcessed);
        
        if (language) {
            formData.append('language', language);
        }

        const response = await this.client.post('/analyze', formData, {
            headers: {
                ...formData.getHeaders(),
            },
        });
        return response.data;
    }

    async analyzeText(text, language = 'en') {
        const params = new URLSearchParams();
        params.append('text', text);
        params.append('language', language);

        const response = await this.client.post('/analyze/text', params, {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
        });
        return response.data;
    }

    async getAllAnalyses(limit = 10, offset = 0) {
        const response = await this.client.get('/analyses', {
            params: { limit, offset },
        });
        return response.data;
    }

    async getAnalysisById(analysisId) {
        const response = await this.client.get(`/analyses/${analysisId}`);
        return response.data;
    }

    async getStatistics() {
        const response = await this.client.get('/statistics');
        return response.data;
    }

    async deleteAnalysis(analysisId) {
        const response = await this.client.delete(`/analyses/${analysisId}`);
        return response.data;
    }

    async gracefulShutdown() {
        const response = await this.client.post('/shutdown');
        return response.data;
    }
}

function printSection(title) {
    console.log('\n' + '='.repeat(60));
    console.log(` ${title}`);
    console.log('='.repeat(60));
}

function printJSON(data, title = 'Response') {
    console.log(`\n${title}:`);
    console.log(JSON.stringify(data, null, 2));
}

async function main() {
    const api = new SentimentAnalysisAPI();

    try {
        // 1. Health Check
        printSection('1. Health Check');
        const health = await api.healthCheck();
        printJSON(health, 'Health Status');

        // 2. System Status
        printSection('2. System Status');
        const status = await api.getSystemStatus();
        printJSON(status, 'System Status');

        // 3. Supported Languages
        printSection('3. Supported Languages');
        const languages = await api.getSupportedLanguages();
        printJSON(languages, 'Supported Languages');

        // 4. Text Analysis Examples
        printSection('4. Text Analysis Examples');

        // English text analysis
        console.log('\nEnglish Text Analysis:');
        const englishText = "I am absolutely delighted with the service! The staff was wonderful and everything exceeded my expectations.";
        const englishResult = await api.analyzeText(englishText, 'en');
        printJSON(englishResult, 'English Text Analysis');

        // Neutral text analysis
        console.log('\nNeutral Text Analysis:');
        const neutralText = "The product arrived on time and was as described.";
        const neutralResult = await api.analyzeText(neutralText, 'en');
        printJSON(neutralResult, 'Neutral Text Analysis');

        // Negative text analysis
        console.log('\nNegative Text Analysis:');
        const negativeText = "I'm very disappointed with the quality. The service was terrible and the product broke immediately.";
        const negativeResult = await api.analyzeText(negativeText, 'en');
        printJSON(negativeResult, 'Negative Text Analysis');

        // 5. Audio Analysis (if sample file exists)
        printSection('5. Audio Analysis');
        const sampleAudio = path.join(__dirname, '..', 'incoming', 'Shoppee_Happy_1.opus');
        if (fs.existsSync(sampleAudio)) {
            try {
                const audioResult = await api.analyzeAudioFile(sampleAudio, 'en');
                printJSON(audioResult, 'Audio Analysis Result');
            } catch (error) {
                console.log(`Audio analysis failed: ${error.message}`);
            }
        } else {
            console.log(`Sample audio file not found: ${sampleAudio}`);
            console.log('Skipping audio analysis example');
        }

        // 6. Get All Analyses
        printSection('6. Get All Analyses');
        const analyses = await api.getAllAnalyses(5);
        printJSON(analyses, 'Recent Analyses');

        // 7. Get Statistics
        printSection('7. System Statistics');
        const stats = await api.getStatistics();
        printJSON(stats, 'System Statistics');

        // 8. Get Specific Analysis (if available)
        if (analyses.analyses && analyses.analyses.length > 0) {
            printSection('8. Get Specific Analysis');
            const firstAnalysisId = analyses.analyses[0].analysis_id;
            const specificAnalysis = await api.getAnalysisById(firstAnalysisId);
            printJSON(specificAnalysis, `Analysis ID ${firstAnalysisId}`);
        }

        // 9. Demonstrate different sentiment examples
        printSection('9. Sentiment Examples');

        const sentimentExamples = [
            {
                text: "This is the best product I've ever used!",
                label: "Very Positive"
            },
            {
                text: "The service was okay, nothing special.",
                label: "Neutral"
            },
            {
                text: "I'm extremely frustrated with this terrible experience.",
                label: "Very Negative"
            }
        ];

        for (const example of sentimentExamples) {
            console.log(`\n${example.label}:`);
            const result = await api.analyzeText(example.text, 'en');
            printJSON(result, `${example.label} Analysis`);
        }

        printSection('Examples Completed Successfully!');
        console.log('\nNote: To test graceful shutdown, uncomment the following line:');
        console.log('// const shutdownResult = await api.gracefulShutdown();');

    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            console.error('Error: Could not connect to the API server.');
            console.error('Make sure the server is running on http://localhost:8000');
        } else {
            console.error('Error during API testing:', error.message);
        }
    }
}

// Handle FormData for Node.js (if not available)
if (typeof FormData === 'undefined') {
    const FormData = require('form-data');
    global.FormData = FormData;
}

if (require.main === module) {
    main();
}

module.exports = SentimentAnalysisAPI; 