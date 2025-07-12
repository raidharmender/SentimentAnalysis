# 🏗️ System Architecture

## 📋 Overview

The Audio Sentiment Analysis System is a comprehensive pipeline that processes audio files to extract customer sentiment insights. The system follows a modular, microservices-inspired architecture with clear separation of concerns.

## 🎯 System Goals

- **Scalability**: Handle multiple concurrent audio processing requests
- **Reliability**: Robust error handling and data persistence
- **Performance**: Optimized processing pipeline with caching
- **Extensibility**: Modular design for easy feature additions
- **User Experience**: Intuitive API and web dashboard

## 🏛️ High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Dashboard] --> B[API Gateway]
        C[Mobile App] --> B
        D[Third-party Integration] --> B
    end
    
    subgraph "API Layer"
        B --> E[FastAPI Server]
        E --> F[Authentication]
        E --> G[Rate Limiting]
        E --> H[Request Validation]
    end
    
    subgraph "Processing Pipeline"
        I[Audio Ingestion] --> J[Audio Preprocessing]
        J --> K[Speech-to-Text]
        K --> L[Sentiment Analysis]
        L --> M[Results Aggregation]
    end
    
    subgraph "Data Layer"
        N[SQLite/PostgreSQL] --> O[Analysis Results]
        P[Redis Cache] --> Q[Session Data]
        R[File Storage] --> S[Audio Files]
    end
    
    subgraph "Monitoring"
        T[Health Checks]
        U[Performance Metrics]
        V[Error Logging]
    end
    
    E --> I
    M --> N
    M --> P
    I --> R
    E --> T
    E --> U
    E --> V
```

## 🔄 Processing Pipeline Architecture

```mermaid
flowchart LR
    subgraph "Input"
        A1[WAV/MP3/FLAC Audio] --> A2[File Validation]
        A2 --> A3[Size Check]
    end
    
    subgraph "Stage 1: Audio Preprocessing"
        B1[Load Audio] --> B2[Sample Rate Normalization]
        B2 --> B3[Channel Conversion]
        B3 --> B4[Noise Reduction]
        B4 --> B5[Silence Trimming]
        B5 --> B6[Amplitude Normalization]
    end
    
    subgraph "Stage 2: Transcription"
        C1[Whisper Model] --> C2[Speech Recognition]
        C2 --> C3[Language Detection]
        C3 --> C4[Confidence Scoring]
        C4 --> C5[Segment Extraction]
    end
    
    subgraph "Stage 3: Sentiment Analysis"
        D1[Hugging Face Model] --> D2[Transformer Analysis]
        D3[VADER] --> D4[Rule-based Analysis]
        D5[TextBlob] --> D6[ML-based Analysis]
        D2 --> D7[Ensemble Aggregation]
        D4 --> D7
        D6 --> D7
        D7 --> D8[Confidence Calculation]
    end
    
    subgraph "Stage 4: Postprocessing"
        E1[Results Aggregation] --> E2[Segment Analysis]
        E2 --> E3[Summary Generation]
        E3 --> E4[Database Storage]
    end
    
    A3 --> B1
    B6 --> C1
    C5 --> D1
    C5 --> D3
    C5 --> D5
    D8 --> E1
    E4 --> F1[API Response]
    E4 --> F2[Dashboard Update]
```

## 🗄️ Data Architecture

```mermaid
erDiagram
    AUDIO_ANALYSES {
        int id PK
        string filename
        string file_path
        int file_size
        float duration
        text transcript
        float transcription_confidence
        string sentiment_label
        float sentiment_score
        json sentiment_details
        float processing_time
        datetime created_at
        datetime updated_at
    }
    
    SPEAKER_SEGMENTS {
        int id PK
        int audio_analysis_id FK
        string speaker_id
        float start_time
        float end_time
        text transcript
        string sentiment_label
        float sentiment_score
        json sentiment_details
        datetime created_at
    }
    
    AUDIO_ANALYSES ||--o{ SPEAKER_SEGMENTS : "has segments"
```

## 🏢 Component Architecture

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Application]
        B[Request Handler]
        C[File Upload Handler]
        D[Response Formatter]
    end
    
    subgraph "Service Layer"
        E[SentimentAnalysisService]
        F[AudioProcessor]
        G[TranscriptionService]
        H[SentimentAnalyzer]
    end
    
    subgraph "Data Access Layer"
        I[Database Manager]
        J[Cache Manager]
        K[File Manager]
    end
    
    subgraph "External Services"
        L[Whisper Model]
        M[Hugging Face Models]
        N[VADER Sentiment]
        O[TextBlob]
    end
    
    subgraph "Infrastructure"
        P[SQLite/PostgreSQL]
        Q[Redis Cache]
        R[File System]
        S[Logging System]
    end
    
    A --> B
    B --> C
    C --> E
    E --> F
    E --> G
    E --> H
    E --> I
    E --> J
    E --> K
    F --> R
    G --> L
    H --> M
    H --> N
    H --> O
    I --> P
    J --> Q
    K --> R
    A --> S
```

## 🔐 Security Architecture

```mermaid
graph TB
    subgraph "Client"
        A[User/Bot]
    end
    
    subgraph "Security Layer"
        B[Rate Limiting]
        C[Input Validation]
        D[File Type Validation]
        E[Size Validation]
        F[Authentication]
        G[Authorization]
    end
    
    subgraph "Application"
        H[API Endpoints]
        I[Business Logic]
        J[Data Access]
    end
    
    subgraph "Data"
        K[Database]
        L[File Storage]
        M[Cache]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    J --> L
    J --> M
```

## 📊 Monitoring & Observability

```mermaid
graph TB
    subgraph "Application Metrics"
        A[Request Count]
        B[Processing Time]
        C[Error Rate]
        D[Success Rate]
    end
    
    subgraph "System Metrics"
        E[CPU Usage]
        F[Memory Usage]
        G[Disk Usage]
        H[Network I/O]
    end
    
    subgraph "Business Metrics"
        I[Analyses Per Day]
        J[Average Sentiment Score]
        K[Model Performance]
        L[User Activity]
    end
    
    subgraph "Monitoring Tools"
        M[Health Checks]
        N[Performance Profiling]
        O[Error Tracking]
        P[Log Aggregation]
    end
    
    A --> M
    B --> N
    C --> O
    D --> P
    E --> M
    F --> N
    G --> O
    H --> P
    I --> M
    J --> N
    K --> O
    L --> P
```

## 🚀 Deployment Architecture

### Development Environment

```mermaid
graph TB
    subgraph "Development"
        A[Local Machine]
        B[Python Virtual Environment]
        C[SQLite Database]
        D[Local File Storage]
    end
    
    A --> B
    B --> C
    B --> D
```

### Docker Environment

```mermaid
graph TB
    subgraph "Docker Compose"
        A[Sentiment Analysis App]
        B[PostgreSQL Database]
        C[Redis Cache]
        D[Nginx Proxy]
    end
    
    subgraph "Volumes"
        E[Upload Directory]
        F[Processed Directory]
        G[Database Data]
        H[Cache Data]
    end
    
    A --> E
    A --> F
    A --> B
    A --> C
    B --> G
    C --> H
    D --> A
```

### Production Environment

```mermaid
graph TB
    subgraph "Load Balancer"
        A[Cloud Load Balancer]
    end
    
    subgraph "Application Tier"
        B[App Instance 1]
        C[App Instance 2]
        D[App Instance N]
    end
    
    subgraph "Data Tier"
        E[Primary Database]
        F[Read Replica]
        G[Redis Cluster]
        H[Object Storage]
    end
    
    subgraph "Monitoring"
        I[Application Monitoring]
        J[Database Monitoring]
        K[Infrastructure Monitoring]
    end
    
    A --> B
    A --> C
    A --> D
    B --> E
    B --> F
    B --> G
    B --> H
    C --> E
    C --> F
    C --> G
    C --> H
    D --> E
    D --> F
    D --> G
    D --> H
    B --> I
    E --> J
    G --> K
```

## 🔄 Data Flow Architecture

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Service
    participant AudioProcessor
    participant TranscriptionService
    participant SentimentAnalyzer
    participant Database
    participant Cache
    
    Client->>API: Upload Audio File
    API->>Service: Process Audio
    Service->>AudioProcessor: Preprocess Audio
    AudioProcessor-->>Service: Processed Audio
    Service->>TranscriptionService: Transcribe Audio
    TranscriptionService-->>Service: Transcription Result
    Service->>SentimentAnalyzer: Analyze Sentiment
    SentimentAnalyzer-->>Service: Sentiment Result
    Service->>Database: Store Results
    Service->>Cache: Cache Results
    Service-->>API: Analysis Complete
    API-->>Client: Return Results
```

## 🎯 Performance Architecture

```mermaid
graph TB
    subgraph "Performance Optimization"
        A[Model Caching]
        B[Result Caching]
        C[Async Processing]
        D[Connection Pooling]
        E[File Streaming]
        F[Compression]
    end
    
    subgraph "Resource Management"
        G[Memory Management]
        H[CPU Optimization]
        I[Disk I/O Optimization]
        J[Network Optimization]
    end
    
    subgraph "Scaling Strategies"
        K[Horizontal Scaling]
        L[Vertical Scaling]
        M[Load Balancing]
        N[Auto-scaling]
    end
    
    A --> G
    B --> H
    C --> I
    D --> J
    E --> K
    F --> L
    G --> M
    H --> N
```

## 🔧 Configuration Architecture

```mermaid
graph TB
    subgraph "Configuration Sources"
        A[Environment Variables]
        B[Configuration Files]
        C[Database Settings]
        D[Runtime Configuration]
    end
    
    subgraph "Configuration Categories"
        E[API Settings]
        F[Audio Processing]
        G[Model Configuration]
        H[Database Settings]
        I[Security Settings]
        J[Monitoring Settings]
    end
    
    subgraph "Configuration Management"
        K[Config Validator]
        L[Config Loader]
        M[Config Cache]
        N[Config Watcher]
    end
    
    A --> K
    B --> L
    C --> M
    D --> N
    K --> E
    L --> F
    M --> G
    N --> H
    K --> I
    L --> J
```

## 🛡️ Error Handling Architecture

```mermaid
graph TB
    subgraph "Error Types"
        A[Validation Errors]
        B[Processing Errors]
        C[System Errors]
        D[External Service Errors]
    end
    
    subgraph "Error Handling"
        E[Input Validation]
        F[Graceful Degradation]
        G[Retry Logic]
        H[Circuit Breaker]
    end
    
    subgraph "Error Response"
        I[Error Logging]
        J[Error Reporting]
        K[User Feedback]
        L[Error Recovery]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
```

## 📈 Scalability Architecture

```mermaid
graph TB
    subgraph "Current State"
        A[Single Instance]
        B[SQLite Database]
        C[Local File Storage]
        D[In-Memory Processing]
    end
    
    subgraph "Scaled State"
        E[Multiple Instances]
        F[PostgreSQL Cluster]
        G[Distributed File Storage]
        H[Distributed Processing]
    end
    
    subgraph "Scaling Strategies"
        I[Load Balancing]
        J[Database Sharding]
        K[CDN Integration]
        L[Message Queues]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
```

## 🔄 API Architecture

```mermaid
graph TB
    subgraph "API Endpoints"
        A[POST /analyze]
        B[GET /analyses]
        C[GET /analyses/{id}]
        D[GET /statistics]
        E[GET /status]
        F[DELETE /analyses/{id}]
        G[GET /health]
    end
    
    subgraph "Request Flow"
        H[Authentication]
        I[Validation]
        J[Processing]
        K[Response]
    end
    
    subgraph "Response Types"
        L[JSON Response]
        M[File Download]
        N[Streaming Response]
        O[Error Response]
    end
    
    A --> H
    B --> I
    C --> J
    D --> K
    E --> L
    F --> M
    G --> N
    H --> O
```

## 🎨 Dashboard Architecture

```mermaid
graph TB
    subgraph "Dashboard Components"
        A[File Upload Interface]
        B[Results Visualization]
        C[Statistics Dashboard]
        D[System Status]
    end
    
    subgraph "Data Sources"
        E[API Endpoints]
        F[Real-time Updates]
        G[Historical Data]
        H[System Metrics]
    end
    
    subgraph "Visualization"
        I[Charts & Graphs]
        J[Interactive Tables]
        K[Real-time Metrics]
        L[Export Features]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
```

## 🔮 Future Architecture Extensions

```mermaid
graph TB
    subgraph "Current Capabilities"
        A[Audio Processing]
        B[Sentiment Analysis]
        C[Basic Dashboard]
        D[API Access]
    end
    
    subgraph "Future Enhancements"
        E[Real-time Streaming]
        F[Multi-language Support]
        G[Advanced Analytics]
        H[Machine Learning Pipeline]
        I[Integration APIs]
        J[Mobile Applications]
    end
    
    subgraph "Advanced Features"
        K[Speaker Diarization]
        L[Emotion Detection]
        M[Topic Modeling]
        N[Custom Model Training]
        O[Advanced Reporting]
        P[Predictive Analytics]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> K
    H --> L
    I --> M
    J --> N
    K --> O
    L --> P
```

## 📋 Architecture Principles

### 1. **Modularity**
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to test and maintain

### 2. **Scalability**
- Horizontal scaling support
- Stateless design where possible
- Efficient resource utilization

### 3. **Reliability**
- Comprehensive error handling
- Data persistence and backup
- Health monitoring and alerting

### 4. **Security**
- Input validation and sanitization
- Authentication and authorization
- Secure data transmission and storage

### 5. **Performance**
- Caching strategies
- Optimized algorithms
- Resource management

### 6. **Maintainability**
- Clear documentation
- Consistent coding standards
- Comprehensive testing

## 🎯 Technology Stack

### **Backend**
- **Python 3.12**: Core programming language
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: Database ORM
- **Pydantic**: Data validation

### **Audio Processing**
- **Librosa**: Audio analysis and processing
- **SoundFile**: Audio file I/O
- **NoiseReduce**: Audio denoising
- **FFmpeg**: Audio format conversion

### **Machine Learning**
- **OpenAI Whisper**: Speech-to-text
- **Hugging Face Transformers**: Sentiment analysis
- **VADER**: Rule-based sentiment analysis
- **TextBlob**: Text processing and sentiment

### **Frontend**
- **Streamlit**: Web dashboard
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

### **Infrastructure**
- **Docker**: Containerization
- **PostgreSQL**: Production database
- **Redis**: Caching and sessions
- **Nginx**: Reverse proxy

### **Monitoring**
- **Health Checks**: System monitoring
- **Logging**: Application logs
- **Metrics**: Performance tracking

This architecture provides a solid foundation for a production-ready sentiment analysis system that can scale with business needs while maintaining high performance and reliability. 