# Theseus Insight

<p align="center">
  <img src="assets/theseus%20insight%20logo.png" alt="Theseus Insight logo" width="300"/>
</p>

Theseus Insight is an end‑to‑end platform for analysing research papers and generating newsletters or podcast episodes from them.  The project provides a FastAPI backend, a React interface and utility scripts that orchestrate language models and text‑to‑speech engines.

## Table of Contents
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Features](#features)
- [LLM Inference Architecture](#llm-inference-architecture)
- [Database Setup](#database-setup)
- [Environment Variables](#environment-variables)
- [Custom Data Storage Location](#custom-data-storage-location)
- [Key Endpoints](#key-endpoints)
  - [Research Timeline](#research-timeline)
  - [Profile Star Map](#profile-star-map)
  - [Research Agent System](#research-agent-system)
  - [Profile Management](#profile-management)
- [Database Migration](#database-migration)
- [Credits](#credits)

---

## Overview

Theseus Insight fetches and ranks papers from [ArXiv](https://arxiv.org/) or provided PDFs, produces newsletters summarising the most relevant papers and can create podcast episodes with optional video visualisations.  A modern React UI communicates with the FastAPI backend via REST and WebSocket endpoints, providing real‑time feedback while background tasks run. 

**🔄 Now Powered by PostgreSQL:** The latest version has been fully migrated from SQLite to PostgreSQL with pgvector, enabling advanced vector similarity search, improved performance, and enhanced scalability for large research paper collections.

**🤖 Unified LLM Interface:** Theseus Insight now uses [LLMFactory](https://github.com/M-Chimiste/LLMFactory.git) for LLM inference, providing seamless integration with multiple providers including OpenAI, Anthropic, Gemini, Ollama, LM Studio, and LlamaCPP with support for local and remote inference.

The system introduces the **Mind-Map Explorer**, an interactive visualization system for exploring multi-order research paper relationships through configurable network graphs, enabling researchers to discover both direct and indirect connections in the academic literature.

**NEW: Research Timeline** - An interactive stream graph visualization that tracks how your research interests evolve over time, with profile integration, multi-level zoom, key paper discovery, and phase classification to help researchers understand trends in their areas of focus.

**NEW: Profile Star Map** - A stunning 3D constellation visualization that displays up to 10,000 papers as an interactive star field, with papers clustered by research interest and color-coded by constellation, enabling researchers to explore their paper collection spatially and discover thematic patterns.

---

## Quickstart

1. **Clone the repository**
   ```bash
   git clone https://github.com/M-Chimiste/TheseusInsight.git
   cd TheseusInsight
   ```

2. **Create a `.env` file** with your API keys (see [Environment Variables](#environment-variables)):
   ```bash
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   # ... other variables as needed
   ```
   > **⚠️ Important:** Do NOT set `DATABASE_URL` in your `.env` file when using Docker Compose. The `docker-compose.yml` automatically configures the correct database connection. Setting it manually will cause connection failures.

3. **Start the stack with Docker** (includes PostgreSQL with pgvector)
   ```bash
   docker compose up --build
   ```
   The API and built React frontend will be available on <http://localhost:8000>.

> **📋 Note:** Theseus Insight requires PostgreSQL with pgvector. SQLite is no longer supported. For local PostgreSQL installations (without Docker), see [Database Setup](#database-setup).

During development you can run the React interface separately:
```bash
cd theseus-ui
npm install
npm run dev
```
This starts Vite on <http://localhost:5173> which proxies API requests to the backend.

### 🔄 Upgrading from SQLite Version?

If you're upgrading from a previous SQLite-based installation of Theseus Insight, you'll need to migrate your data to PostgreSQL. The process is automated and preserves all your papers, embeddings, and generated content:

📖 **[Complete Migration Guide](docs/migration_guide.md)** - Step-by-step migration instructions with examples.

**Quick Migration:**
```bash
# 1. Export your SQLite data
python -m theseus_insight.utils.db_migration.db_export --source-db data/theseus.db --output backup.json

# 2. Set up PostgreSQL and update your .env file
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/theseus

# 3. Import to PostgreSQL
python -m theseus_insight.utils.db_migration.db_import --input backup.json
```

---

## Features

- **FastAPI** server with endpoints for paper management, newsletter and podcast pipelines and visualiser generation.
- **React** frontend built with Vite and Material UI, served from the backend in production.
- **Real‑time progress** streaming over WebSockets for long running tasks.
- **PostgreSQL database** (with pgvector) for storing papers, runs and configuration data with advanced vector similarity search and full-text indexing.
- **Advanced hybrid search capabilities** combining semantic similarity via vector embeddings with keyword precision using PostgreSQL's native full-text search - significantly more powerful than traditional SQLite-based approaches.
- **Robust ArXiv Data Access**: Automatic fallback from live ArXiv OAI-PMH API to Kaggle dataset (1.7M+ papers) with auto-download and cleanup when the API is unavailable.
- **Cached Summaries & Keywords**: LLM-generated paper summaries and YAKE-extracted top keywords are stored in the database, eliminating redundant generation and dramatically speeding up subsequent mind-map builds.
- **Mind-Map Explorer**: An interactive visualization tool to explore the intellectual neighborhood of research papers.
  - **Multi-Order Expansion**: Generate mind-maps with configurable expansion orders (1-5) to explore deeper connections between papers.
  - **Visual Network Exploration**: Interactive force-directed layouts with similarity-based node positioning and edge weighting.
  - **Configurable Parameters**: Adjust similarity thresholds (10%-80%), paper count (5-30), and nodes per expansion order (5-50).
  - **Smart Deduplication**: Automatic duplicate detection and removal across expansion orders for clean visualizations.
  - **Real-Time Progress Tracking**: WebSocket-powered progress updates with detailed step information during generation.
  - **Save & Share Reports**: Save mind-map configurations and results for future reference and collaboration.
  - **Seamless Integration**: Launch from any paper in the research library or similarity search results.
- **Research Timeline**: Interactive stream graph visualization for tracking research interest evolution over time.
  - **Stream Graph Visualization**: Beautiful D3.js-powered flowing stream graphs showing how your research interests evolve over time with vibrant color-coded streams.
  - **Profile-Integrated Tracking**: Automatically tracks your profile's research interests with configurable interest selection and multi-profile support.
  - **Multi-Level Zoom**: Explore at year, quarter, month, or week granularity with smooth transitions between zoom levels.
  - **Key Paper Discovery**: Hover over any time period to see the most influential papers for each research interest during that period.
  - **Phase Indicators**: Visual phase classification (Emerging, Growth, Stable, Declining) based on growth rate analysis with color-coded legend.
  - **AI Short Labels**: One-click generation of concise AI-summarized labels for long research interest descriptions.
  - **Click-Through Navigation**: Click any point on the timeline to navigate directly to filtered paper results for that research interest and time period.
  - **Export to Papers**: Seamless integration with the paper library for deep-diving into specific time periods and interests.
- **Profile Star Map**: Interactive 3D constellation visualization of your research paper collection.
  - **3D Star Field**: Papers rendered as stars in a 3D space using dimensionality reduction of paper embeddings, preserving semantic relationships.
  - **Constellation Clustering**: Papers are grouped by dominant research interest with gentle gravitational clustering, creating organic "constellation" formations.
  - **Color-Coded Constellations**: Each research interest is assigned a unique color, making it easy to identify thematic clusters at a glance.
  - **Interactive Controls**: Click and drag to rotate in 3D, scroll to zoom, with smooth animations and depth-aware rendering (near stars appear larger/brighter).
  - **Centroid Markers**: Glowing constellation centroids with labeled background pills show the "center of mass" for each research interest.
  - **Paper Details on Click**: Click any star to see paper details including title, date, and dominant research interest, with direct navigation to the Papers page.
  - **2D/3D Toggle**: Switch between 2D and 3D visualization modes based on preference.
  - **Dashboard Preview**: Auto-rotating 3D preview widget on the dashboard with drag-to-rotate interaction.
  - **Insights Panel**: Quick stats showing total papers, unlabeled count, and top constellations with paper counts.
  - **Recompute on Demand**: Trigger star map recalculation after adding new papers or updating research interests.
- **Comprehensive Profile Management System**: Create and manage multiple research profiles with individual configurations.
  - **Smart Profile Selection**: Visual profile chips with individual remove buttons and system-wide Smart Selection Bar for easy multi-profile workflows.
  - **Profile-Specific Configurations**: Each profile maintains its own email recipients, research interests, ArXiv category filters, tags, and visual styling.
  - **Multi-Profile Support**: Select multiple profiles simultaneously with combined email recipients and research interests while maintaining backward compatibility.
  - **Profile Integration**: Seamless integration across all features - newsletters, podcasts, mind-maps, and research agent workflows use profile-specific settings.
  - **Default Profile System**: Designate default profiles for streamlined workflows with automatic population of settings for new profiles.
  - **Visual Organization**: Color-coded profiles with custom descriptions and tag-based organization for easy identification.
- **Unified LLM Interface via LLMFactory**: Powered by [LLMFactory](https://github.com/M-Chimiste/LLMFactory.git), providing a consistent interface across multiple inference providers including OpenAI, Anthropic, Gemini, Ollama, LM Studio, and LlamaCPP with support for streaming, structured output, and multimodal capabilities.
- **Multi-Server Inference**: Distribute LLM workloads across multiple Ollama or LM Studio servers for parallel processing.
  - **Bulk Judge Operations**: Score large paper collections using distributed worker pools
  - **Newsletter Multi-Server Scoring**: Accelerate newsletter paper scoring with parallel inference across servers
  - **Per-Server Model Configuration**: Configure different models and parameters for each server in non-homogeneous deployments
  - **Real-Time Progress Tracking**: WebSocket-powered updates with per-server statistics and task breakdown
- **Flexible TTS providers** including OpenAI, Amazon Polly, and KokoroTTS.
- **Encrypted credential storage** with a UI for managing API keys in Settings.
- **Dockerfile and Compose setup** to run the entire application in containers with PostgreSQL.
- **Advanced Research Agent System**: Dual-mode AI research orchestration with support for both single-agent sequential workflows and multi-agent parallel processing.
  - **Single-Agent Mode**: Sequential workflow with iterative research loops, evidence gathering, and adaptive compression for deep analysis.
  - **Multi-Agent Mode**: Parallel orchestration with specialized agents (Research, Analysis, Verification, Alternative Perspectives) for comprehensive coverage.
  - **Structured Output Support**: Direct structured output for compatible providers (Ollama, LlamaCPP) with automatic fallback parsing for cloud APIs.
  - **Intelligent Model Routing**: Automatic model selection with provider-specific optimizations and rate limiting for local hardware.
  - **Sequential Local Model Queuing**: Thread-safe queuing system ensures local Ollama/LlamaCPP models run sequentially to prevent resource conflicts.
  - **Real-Time Progress Tracking**: WebSocket-powered progress updates with detailed agent status and execution metrics.
  - **Configuration Management**: Comprehensive settings interface for model selection, workflow parameters, and agent specialization.
  - **Research History**: Complete research run tracking with results persistence and analysis capabilities.

---

## LLM Inference Architecture

Theseus Insight leverages **[LLMFactory](https://github.com/M-Chimiste/LLMFactory.git)**, a unified inference library that provides a consistent interface across multiple LLM providers. This architecture enables seamless switching between local and cloud-based models without changing application code.

### Supported LLM Providers

**Local Inference:**
- **Ollama** - Local model inference with support for Llama, Mistral, Qwen, and other open models
- **LM Studio** - Local inference with remote connection support, configurable context length, and GPU offload options
- **LlamaCPP** - Direct GGUF model loading with hardware acceleration

**Cloud APIs:**
- **OpenAI** - GPT-4, GPT-4o, GPT-3.5-turbo models
- **Anthropic** - Claude models via direct API or AWS Bedrock
- **Google Gemini** - Gemini Pro and Flash models
- **Custom OpenAI-Compatible** - Any OpenAI-compatible API endpoint

**Embedding Models:**
- **Sentence Transformers** - HuggingFace embedding models with GPU acceleration
- **Ollama Embeddings** - Local embedding generation via Ollama

### Key Features

- **Unified Interface**: Single API for all providers with consistent message format
- **Streaming Support**: Token-by-token streaming across all providers
- **Structured Output**: Schema-based JSON output with Pydantic models for compatible providers
- **Multimodal Support**: Vision capabilities for compatible models
- **Flexible Configuration**: Environment variables, direct parameters, or database-stored credentials
- **Type Safety**: Full type hints for better IDE support and error detection

### LM Studio Configuration

LM Studio provides local LLM inference with enterprise features:

**Connection Options:**
```bash
# Local default
LMSTUDIO_HOST=localhost:1234

# Remote connection
LMSTUDIO_HOST=athena.local:1234

# Disable Qwen thinking mode in LM Studio requests
LMSTUDIO_DISABLE_THINKING=true
```

**Advanced Features:**
- **Remote Connections**: Connect to LM Studio instances on other machines
- **Context Length Configuration**: Customize context windows (e.g., 32768, 131072 tokens)
- **GPU Offload Options**: Configure GPU memory usage (`max`, `off`, or ratio 0-1)
- **Model Hot-Swapping**: Change models without restarting the application

**Qwen Thinking Control:**
- Theseus disables thinking by default for Qwen models served through LM Studio.
- This behavior is controlled by `LMSTUDIO_DISABLE_THINKING` and currently defaults to `true`.
- When enabled, Theseus forces `use_thinking=False` and injects Qwen's `/no_think` directive into the request as a fallback.
- Set `LMSTUDIO_DISABLE_THINKING=false` if you want to allow Qwen thinking mode again.

**Newsletter PDF Extraction Safety:**
- Newsletter section generation now tries Docling first for PDF-to-markdown extraction, then falls back to MarkItDown if Docling fails or times out.
- Each parser runs in its own subprocess with a hard timeout so a corrupted or wedged PDF cannot block the whole newsletter pipeline.
- The timeout is controlled by `PDF_CONVERSION_TIMEOUT_SEC` and defaults to `120` seconds.

**SMTP DNS Fallback:**
- Gmail delivery now retries hostname resolution against fallback DNS servers if the local system resolver fails, which is helpful on unstable VPN/public-network combinations.
- Fallback DNS is enabled by default and tries Google Public DNS over both TCP and UDP (`tcp://8.8.8.8,tcp://8.8.4.4,8.8.8.8,8.8.4.4`) unless overridden.
- You can tune this with `SMTP_DNS_FALLBACK_ENABLED`, `SMTP_DNS_NAMESERVERS`, `SMTP_DNS_TIMEOUT_SEC`, and `SMTP_CONNECT_TIMEOUT_SEC`.

**Gmail API Delivery Fallback:**
- Newsletter email delivery now tries Gmail SMTP first, then falls back to the Gmail API over HTTPS if SMTP fails.
- The Gmail API fallback requires a saved OAuth token file with the `gmail.send` scope.
- Use `scripts/send_test_email.py --send --authorize` once to create the token file locally.
- The token file defaults to `gmail_token.json`, and the Gmail API request timeout defaults to `30` seconds.

**Newsletter Intro Timeout:**
- The newsletter intro generation step can use a much longer LM Studio request timeout than the rest of the pipeline.
- This is controlled by `NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC` and defaults to `1200` seconds.
- Set `NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC=0` (or `none`) to disable the timeout entirely and let the intro request run as long as needed.
- This setting is only applied to the newsletter intro model when that model uses `lmstudio`.

**Example Configuration** (in `config/orchestration.json`):
```json
{
  "judge_model": {
    "model_name": "granite-4.0-h-tiny-mlx",
    "model_type": "lmstudio",
    "max_new_tokens": 512,
    "temperature": 0.2,
    "num_ctx": 128000,
    "host": null
  }
}
```

When `host` is `null`, LM Studio uses the `LMSTUDIO_HOST` environment variable or defaults to `localhost:1234`.

### Multi-Server Support

Theseus Insight supports configuring multiple inference servers for distributed processing:
- **Multiple Ollama Servers**: Configure multiple Ollama instances for parallel processing
- **Multiple LM Studio Servers**: Distribute workload across multiple LM Studio instances
- **Dynamic Load Balancing**: Automatic task distribution across available servers
- **Health Monitoring**: Server health checks and automatic failover

Configure additional servers in Settings → Inference Servers.

### Advanced Configuration: Per-Model Host and Non-Homogeneous Deployments

Theseus Insight supports advanced deployment scenarios with flexible host configuration and per-server model customization:

#### Per-Model Host Configuration

Configure different hosts for different models or tasks using a 3-tier priority system:

**Priority System:**
1. **Configured Host** (Highest Priority) - Set directly in model configuration
2. **Environment Variable** - Falls back to `OLLAMA_URL` or `LMSTUDIO_HOST`
3. **Provider Default** (Lowest Priority) - Uses default localhost settings

**Example Use Cases:**

**Newsletter on Remote Server, Bulk Judge on Local:**
```json
{
  "newsletter_intro_model": {
    "model_name": "qwen2.5:32b",
    "model_type": "ollama",
    "host": "athena.local:11434"
  },
  "judge_model": {
    "model_name": "phi4:latest",
    "model_type": "ollama",
    "host": null
  }
}
```

**Multiple Remote Hosts:**
```json
{
  "research_agent_model_config": {
    "boss_model": {
      "model_name": "qwen2.5:32b",
      "model_type": "ollama",
      "host": "server1.local:11434"
    }
  },
  "podcast_model": {
    "model_name": "granite-4.0-h-tiny-mlx",
    "model_type": "lmstudio",
    "host": "server2.local:1234"
  }
}
```

**Configuration via UI:** Set the "Host (Optional)" field in Settings → Model Configuration for any Ollama, LMStudio, or Custom-OAI model.

#### Non-Homogeneous Multi-Server Deployments

Configure different models and parameters for each inference server in distributed bulk processing:

**Per-Server Model Override:**
- Configure unique model names for each server (e.g., "phi4:latest" on Ollama, "phi4-mlx" on LMStudio)
- Set per-server model parameters (temperature, max_new_tokens, context window)
- Mix different model variants across servers while maintaining API compatibility

**Configuration Example:**

| Server | Provider | Model | Parameters | Use Case |
|--------|----------|-------|------------|----------|
| GPU Server 1 | Ollama | `qwen2.5:32b` | `temperature: 0.1, num_ctx: 131072` | High-quality analysis |
| GPU Server 2 | LMStudio | `phi4-mlx` | `temperature: 0.2, context_length: 128000` | Fast processing |
| CPU Server | Ollama | `phi4:latest` | `temperature: 0.3, num_ctx: 32768` | Fallback/testing |

**Benefits:**
- **Hardware Optimization**: Use quantized models (MLX) on Apple Silicon, full models on NVIDIA GPUs
- **Cost Efficiency**: Deploy cheaper models on some servers, premium models on others
- **A/B Testing**: Compare different models/parameters across servers in production
- **Graceful Degradation**: Fall back to lighter models on resource-constrained servers

**Configuration via UI:**
1. Navigate to Settings → Inference Servers
2. Add or edit a server
3. Set "Model Name (optional)" to override the global model
4. Expand "Model Configuration Overrides" to set per-server parameters
5. The system automatically uses these overrides for bulk judge operations

**Example Workflow:**
```bash
# Server configurations (via UI)
Server 1 (Ollama): model_name="qwen2.5:32b", temperature=0.1
Server 2 (LMStudio): model_name="phi4-mlx", temperature=0.2
Server 3 (Ollama): NULL (uses global config)

# When bulk judge operation runs:
# - Server 1 uses qwen2.5:32b with temperature 0.1
# - Server 2 uses phi4-mlx with temperature 0.2
# - Server 3 uses global judge_model configuration
```

**Model Validation:**
Use the validation endpoint to verify model availability before deployment:
```bash
POST /api/settings/inference-servers/{server_id}/validate-model
{
  "model_name": "phi4:latest",
  "model_config": {"temperature": 0.2}
}
```

#### Newsletter Multi-Server Scoring

Newsletter generation now supports multi-server LLM judge scoring for faster paper relevance evaluation. When generating newsletters with large paper sets, the scoring workload can be distributed across multiple inference servers.

**Enable via API:**
```json
POST /api/actions/run-newsletter-pipeline
{
  "start_date": "2024-01-01",
  "end_date": "2024-01-07",
  "research_interests": "machine learning, NLP",
  "use_multi_server_judge": true,
  "judge_server_ids": [1, 2, 3],
  "judge_request_timeout_sec": 120,
  "judge_max_retries": 3
}
```

**Features:**
- **Parallel Scoring**: Papers are scored concurrently across selected inference servers
- **Profile-Specific Scoring**: Each paper is scored against each selected profile's research interests
- **Progress Tracking**: Real-time WebSocket updates with per-server statistics and task breakdown
- **Dynamic Load Balancing**: Tasks are distributed via a queue-based system for optimal server utilization
- **Aggregated Scores**: Final paper rankings aggregate scores across all selected profiles

**Monitoring Views:**
The system provides PostgreSQL views for monitoring multi-server newsletter jobs:
- `newsletter_scoring_progress`: Overall job progress with task status breakdown
- `newsletter_server_stats`: Per-server performance metrics (tasks completed, avg duration)

For more details on LLMFactory capabilities, visit the [LLMFactory GitHub repository](https://github.com/M-Chimiste/LLMFactory.git).

---

## Database Setup

Theseus Insight uses **PostgreSQL 14+ with pgvector** as its primary database system, providing significant advantages over the previous SQLite-based architecture:

**✨ PostgreSQL Benefits:**
- **Advanced Vector Search**: Native pgvector extension for efficient semantic similarity
- **Full-Text Search**: Built-in PostgreSQL search with stemming and ranking
- **Hybrid Search**: Combine semantic and keyword search for superior accuracy
- **Scalability**: Handle large research paper collections with excellent performance
- **Concurrent Access**: Multiple users and processes can access the database simultaneously
- **ACID Compliance**: Robust data integrity and transaction support

You have several PostgreSQL setup options:

### Option 1: Docker Compose (Recommended)
PostgreSQL is automatically configured when using Docker Compose:

```bash
docker compose up --build
```

### Option 2: Local PostgreSQL Installation
For development or production installations:

📖 **[Local PostgreSQL Setup Guide](docs/postgresql_setup.md)** - Detailed instructions for installing and configuring PostgreSQL locally.

### Option 3: Docker PostgreSQL Only
To run just PostgreSQL in Docker while running the application locally:

📖 **[Docker PostgreSQL Guide](docs/docker_postgresql.md)** - Instructions for running PostgreSQL in Docker.

### Database Schema Management

Theseus Insight includes an automatic database migration system that ensures your PostgreSQL schema is always up-to-date:

📖 **[Database Migrations Guide](docs/database_migrations.md)** - Learn about the automatic migration system that runs on startup.

**Key Features:**
- **Automatic Schema Updates**: Migrations run automatically when the API starts
- **Version Tracking**: Applied migrations are tracked to prevent re-running
- **Failure Protection**: Migration failures are handled gracefully without corrupting data
- **Profile Support**: Automatically creates database tables for the profiles feature

### Migrating from SQLite
If you're upgrading from a previous SQLite installation:

📖 **[Migration Guide](docs/migration_guide.md)** - Step-by-step instructions for migrating from SQLite to PostgreSQL.

---

## Environment Variables

Create a `.env` file in the project root containing keys and settings:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL connection string (default: `postgresql://postgres:postgres@localhost:5432/theseus`) |
| `OPENAI_API_KEY` | API key for OpenAI models and TTS |
| `ANTHROPIC_API_KEY` | API key for Anthropic Claude models |
| `GOOGLE_API_KEY` | API key for Google Gemini models |
| `OLLAMA_URL` | Base URL of a local Ollama server (default `http://127.0.0.1:11434`) |
| `OLLAMA_PASSTHROUGH` | When `true` (default), Docker containers redirect localhost Ollama URLs to host machine. Set to `false` to use container-local Ollama installation |
| `LMSTUDIO_HOST` | LM Studio server host:port for local inference (default `localhost:1234`). Supports remote connections (e.g., `athena.local:1234`) |
| `LMSTUDIO_DISABLE_THINKING` | Disable Qwen thinking mode for LM Studio requests. Defaults to `true`; set to `false` to allow thinking again |
| `PDF_CONVERSION_TIMEOUT_SEC` | Per-PDF timeout for newsletter section extraction. Defaults to `120`; timed-out or corrupted PDFs are skipped so the run can continue |
| `SMTP_DNS_FALLBACK_ENABLED` | Enable fallback DNS resolution for SMTP hostname lookups when system DNS fails. Defaults to `true` |
| `SMTP_DNS_NAMESERVERS` | Comma-separated DNS servers used by the SMTP fallback resolver. Supports `tcp://` or `udp://` prefixes; defaults to `tcp://8.8.8.8,tcp://8.8.4.4,8.8.8.8,8.8.4.4` |
| `SMTP_DNS_TIMEOUT_SEC` | Timeout per fallback DNS query in seconds. Defaults to `5` |
| `SMTP_CONNECT_TIMEOUT_SEC` | SMTP socket connect timeout in seconds. Defaults to `20` |
| `GMAIL_TOKEN_FILE` | Path to the saved Gmail API OAuth token used when SMTP delivery fails. Defaults to `gmail_token.json` |
| `GMAIL_API_TIMEOUT_SEC` | Timeout for Gmail API send requests in seconds. Defaults to `30` |
| `NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC` | LM Studio request timeout for the newsletter intro generation step. Defaults to `1200`; set to `0` or `none` to disable the timeout entirely |
| `GMAIL_SENDER_ADDRESS` | Gmail address used to send newsletters |
| `GMAIL_APP_PASSWORD` | Gmail App password for SMTP authentication see: [Gmail App Password Instructions](https://support.google.com/mail/answer/185833?hl=en)|
| `DEBUG` | When set to `true`, `1`, or `yes` globally re-enables verbose `print()` statements and DEBUG-level logger output. Leave unset (default) for a quiet console |
| `CLIENT_ID`, `PROJECT_ID`, `CLIENT_SECRET`, `REDIRECT_URI` | OAuth credentials for the YouTube upload helper |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `REGION_NAME` | Credentials for Amazon Polly TTS |
| `PRODUCTION_FRONTEND_URL` | Allowed origin for CORS when deploying the frontend |
| `RUNNING_IN_DOCKER` | Set to `true` in Docker images for correct static file paths |
| `APP_SECRET_KEY` | Secret used to encrypt API credentials stored in the database |

### PostgreSQL Connection Examples

**Local PostgreSQL:**
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/theseus_insight
```

**Docker PostgreSQL:**
```bash
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/theseus
```

**Cloud PostgreSQL (e.g., AWS RDS):**
```bash
DATABASE_URL=postgresql://username:password@hostname:5432/database_name
```

### Manual Installation

If you prefer to install manually or the automated scripts don't work for your environment:

#### Prerequisites
- **Python 3.10+** - Download from [python.org](https://www.python.org/downloads/)
- **Node.js 16+** - Download from [nodejs.org](https://nodejs.org/)
- **PostgreSQL 14+** - See [Database Setup](#database-setup) for installation options

#### Steps
1. **Install Python dependencies**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate.bat
   
   # Install dependencies
   pip install -r requirements.lock
   ```

2. **Install and build frontend**
   ```bash
   cd theseus-ui
   npm install
   npm run build
   cd ..
   ```

3. **Set up PostgreSQL database** (see [Database Setup](#database-setup))

4. **Create data directories**
   ```bash
   mkdir -p data/{newsletters,podcasts,visualizations,temp}
   mkdir -p config
   ```

5. **Configure environment variables** (see [Environment Variables](#environment-variables) section)

6. **Start the application**
   ```bash
   # Backend (in one terminal)
   source venv/bin/activate
   uvicorn theseus_insight.main:app --host 0.0.0.0 --port 8000 --reload
   
   # Frontend (in another terminal) 
   cd theseus-ui
   npm run dev
   ```

The PostgreSQL database schema will be created automatically on first run through the automatic migration system.

## Custom Data Storage Location

By default, Theseus Insight stores generated files (newsletters, podcasts, visualizations) in the local `./data` directory, while the PostgreSQL database is hosted separately. For installations with limited internal storage, you can redirect generated content to an external drive or custom location.

### Quick Setup for External Storage

**Option 1: Using the helper script (recommended)**
```bash
# Start with data stored on external drive
./scripts/start-with-external-storage.sh

# Or specify a custom path
./scripts/start-with-external-storage.sh /path/to/your/storage

# Run in detached mode
./scripts/start-with-external-storage.sh /Volumes/nyx/theseus_insight_data -d
```

**Option 2: Manual Docker Compose override**
```bash
# Set the storage path
export EXTERNAL_DATA_PATH=/Volumes/nyx/theseus_insight_data

# Start with external storage configuration
docker-compose -f docker-compose.yml -f docker-compose.external-storage.yml up --build
```

**Option 3: Environment variable in .env file**
```bash
# Add to your .env file
EXTERNAL_DATA_PATH=/Volumes/nyx/theseus_insight_data

# Then start with the external storage override
docker-compose -f docker-compose.yml -f docker-compose.external-storage.yml up --build
```

### Storage Structure

When using external storage, the following directory structure is created:

```
/Volumes/nyx/theseus_insight_data/
├── app_data/                    # Application-generated files
│   ├── newsletters/             # Generated newsletters
│   ├── podcasts/               # Generated podcast audio/video
│   ├── visualizations/         # Generated visualizations
│   └── temp/                   # Temporary processing files
└── (PostgreSQL data is stored separately via DATABASE_URL)
```

**Note:** Unlike the previous SQLite-based system, PostgreSQL database files are not stored in the application data directory but are managed by the PostgreSQL server according to your `DATABASE_URL` configuration.

#### Hybrid Search Functionality
Theseus Insight features advanced hybrid search capabilities that combine the precision of PostgreSQL's full-text search with the contextual understanding of semantic similarity using pgvector. This provides significantly enhanced search accuracy compared to traditional keyword-only or semantic-only approaches.

**Key Features:**
- **PostgreSQL Full-Text Search**: Uses native `tsvector` and `tsquery` for accurate term frequency scoring with stemming and ranking
- **pgvector Semantic Search**: Leverages PostgreSQL's pgvector extension for efficient cosine similarity search across paper embeddings
- **Dual-Mode Search**: Simultaneously performs semantic similarity using vector embeddings and advanced keyword ranking across paper titles and abstracts
- **Weighted Scoring**: User-adjustable weights for combining semantic and keyword scores (default: 60% semantic, 40% keyword)
- **Real-Time Scoring**: Returns individual `semantic_score`, `keyword_score`, and combined `hybrid_score` for transparency
- **Full Integration**: Works seamlessly with existing filters (date range, score threshold, pagination)
- **Performance Optimized**: Database-level operations use pgvector indexes and PostgreSQL's GIN indexes for scalable performance

**Technical Implementation:**
- **PostgreSQL Full-Text Search**: Utilizes native `tsvector`/`tsquery` with English language configuration and stemming
- **Ranking Algorithm**: Uses `ts_rank()` function for weighted relevance scoring
- **Vector Operations**: Employs pgvector's cosine distance operator (`<=>`) for semantic similarity
- **Index Optimization**: GIN indexes for full-text search and IVFFlat indexes for vector operations
- **Automatic Schema**: PostgreSQL schema automatically includes search vectors and indexes

**Performance Optimized**: Database-level operations use pgvector for vector search and PostgreSQL's native full-text search indexes for scalable performance

### Mind-Map Explorer

**Technical Implementation:**
- **PostgreSQL Integration**: Complete CRUD operations for mind-map reports with JSON storage for configurations and results
- **Vector Similarity**: Uses pgvector for efficient similarity calculations across large paper collections
- **Performance Optimization**: Leverages PostgreSQL's indexing for fast paper retrieval and similarity computation

---

## Key Endpoints

### Research Timeline

The Research Timeline provides an interactive stream graph visualization for tracking how your research interests evolve over time. This feature integrates with the profile system to provide personalized tracking of research areas.

#### Core Timeline Endpoints

**`GET /api/trends/timeline-data`** - Get timeline data for stream graph visualization
- **Parameters**: 
  - `topic_ids` - Comma-separated list of research interest IDs to include
  - `profile_ids` - Comma-separated list of profile IDs to filter by
  - `period_type` - Time granularity (week/month/quarter)
  - `start_date`, `end_date` - Optional date range filters
  - `include_key_papers` - Include influential papers for each period (default: true)
  - `key_papers_limit` - Number of key papers per topic per period
  - `limit` - Maximum number of topics to return
  - `source` - Data source (`profile_interests` for profile-specific interests)
- **Returns**: Timeline data with topics, periods, paper counts, growth rates, and key papers
- **Features**: Phase classification (emerging/growth/stable/declining), period-by-period metrics

**`POST /api/trends/generate-short-labels`** - Generate AI-summarized short labels
- **Parameters**: `profile_ids` - Optional list of profile IDs to generate labels for
- **Returns**: Processing status with count of labels generated
- **Features**: Uses configured LLM model, caches results in database

**`GET /api/profiles/{profile_id}/research-interests`** - Get research interests for timeline
- **Returns**: Profile's research interests with metadata for timeline selection
- **Features**: Returns interest descriptions, similarity thresholds, and associated paper counts

#### Timeline Visualization Features

**Stream Graph Design**:
- **Flowing Streams**: Each research interest is rendered as a smooth, flowing stream whose height represents paper count over time
- **Color Coding**: Vibrant, distinct colors for each research interest (up to 10 unique colors with automatic cycling)
- **Stacked Layout**: D3.js stack layout with wiggle offset for aesthetically pleasing flow

**Interactivity**:
- **Hover Tooltips**: Detailed information including paper count, growth rate, phase, and key papers
- **Click Navigation**: Click any point to navigate to filtered paper results
- **Zoom Controls**: Switch between year/quarter/month/week views
- **Topic Selection**: Multi-select dropdown to filter which interests to display

**Phase Classification**:
- **Emerging** (>50% growth): High-growth areas marked in green
- **Growth** (>10% growth): Growing areas marked in blue  
- **Stable** (±10%): Stable areas marked in gray
- **Declining** (<-10% growth): Declining areas marked in red

**Key Paper Integration**:
- Each time period shows the most relevant papers for that research interest
- Papers are ranked by similarity score to the research interest
- Direct links to paper details from the tooltip

### Profile Star Map

The Profile Star Map provides an immersive 3D visualization of your research paper collection, rendering up to 10,000 papers as an interactive constellation where spatial proximity reflects semantic similarity.

#### Core Star Map Endpoints

**`GET /api/profiles/{profile_id}/star-map`** - Get cached star map points
- **Parameters**: 
  - `limit` - Maximum number of points to return (default: 10000)
- **Returns**: Star map points with coordinates, paper metadata, and constellation assignments
- **Features**: Returns pre-computed 3D coordinates, dominant research interest labels, and centroid positions

**`GET /api/profiles/{profile_id}/star-map/status`** - Get star map computation status
- **Returns**: Computation timestamp, total points, and profile ID
- **Features**: Check if star map needs recomputation after data changes

**`POST /api/profiles/{profile_id}/star-map/recompute`** - Trigger star map recomputation
- **Parameters**: `limit` - Maximum papers to include (default: 10000)
- **Returns**: Task ID for background processing with WebSocket progress tracking
- **Features**: Async computation with real-time progress updates via WebSocket

#### Star Map Visualization Features

**3D Rendering**:
- **Perspective Projection**: Papers rendered with depth-aware sizing (near stars larger, far stars smaller)
- **Rotation Controls**: Click and drag to rotate the view in 3D space
- **Zoom Controls**: Scroll to zoom in/out with configurable limits
- **Depth Ordering**: Back-to-front rendering for correct visual layering

**Constellation System**:
- **Semantic Clustering**: Papers grouped by dominant research interest using embedding-based projection
- **Gravitational Layout**: Papers gently pulled toward their constellation centroid while preserving local semantic structure
- **Color Coding**: Each constellation assigned a unique color from a vibrant palette
- **Centroid Markers**: Glowing markers at constellation centers with labeled background pills

**Interactivity**:
- **Hover Tooltips**: Paper title, date, and research interest on hover
- **Click Navigation**: Click any paper to open details dialog with "Open in Papers" action
- **2D/3D Toggle**: Switch between flat and perspective views
- **Dashboard Preview**: Mini auto-rotating 3D preview with drag interaction

**Technical Implementation**:
- **Dimensionality Reduction**: Random projection from high-dimensional embeddings to 3D coordinates
- **PostgreSQL Storage**: Cached coordinates stored in `profile_star_map_points` table
- **Canvas Rendering**: Hardware-accelerated 2D canvas with custom 3D projection
- **Quadtree Indexing**: Efficient point picking using D3 quadtree for hover/click detection

### Research Agent System

The Research Agent system provides AI-powered research orchestration with support for both single-agent sequential workflows and multi-agent parallel processing. The system automatically routes between modes based on configuration and provides comprehensive research capabilities.

#### Core Research Endpoints

**`POST /api/research-agent/research`** - Start a research task
- **Parameters**: `research_question`, `mode` (single/multi), `config` (optional overrides), `save_to_library`
- **Returns**: Task ID for background processing with WebSocket progress tracking
- **Features**: Automatic mode detection, configurable parameters, research library integration
- **Example**: Start multi-agent research with custom search limits

**`GET /api/research-agent/tasks/{task_id}`** - Get research task status
- **Returns**: Task status, progress information, agent details, execution metrics
- **Features**: Real-time progress updates, detailed agent status for multi-agent mode

**`GET /api/research-agent/tasks/{task_id}/result`** - Get research results
- **Returns**: Final answer, evidence sources, sub-queries, workflow messages, statistics
- **Features**: Comprehensive research output with sources, evidence, and synthesis details

**`GET /api/research-agent/history`** - List research task history
- **Parameters**: `limit`, `offset`, `status_filter`
- **Returns**: Paginated list of research tasks with metadata and statistics
- **Features**: Search and filtering capabilities, execution time analysis

#### Mode Configuration Endpoints

**`GET /api/research-agent/modes`** - Get current research mode configuration
- **Returns**: Current mode, single-agent config, multi-agent config, validation status
- **Features**: Configuration validation, available modes list

**`POST /api/research-agent/modes`** - Switch research agent mode
- **Parameters**: `mode` (single/multi)
- **Returns**: Updated mode configuration with validation results
- **Features**: Automatic configuration validation, mode switching

**`PUT /api/research-agent/config/{mode}`** - Update mode-specific configuration
- **Parameters**: Mode configuration object (models, parameters, thresholds)
- **Returns**: Updated configuration with validation results
- **Features**: Model validation, parameter validation, provider compatibility checks

#### Research Agent Features

**Single-Agent Mode**: Sequential workflow with research loops
- **Boss Model**: Primary model that orchestrates the entire workflow
- **Node-Specific Models**: Optional specialized models for Query Planning, Evidence Selection, Compression, Answer Generation
- **Iterative Research**: Multiple research loops with evidence gathering and adaptive compression
- **Context Management**: Automatic compression when token limits are exceeded

**Multi-Agent Mode**: Parallel orchestration with specialized agents
- **Boss Model**: Orchestrates question generation and final synthesis
- **Specialized Agents**: Research, Analysis, Verification, Alternative Perspective agents
- **Question Generation**: AI-powered decomposition of research questions into specialized sub-questions
- **Parallel Execution**: Concurrent agent execution with progress tracking
- **Synthesis Engine**: Conflict detection and resolution with comprehensive final synthesis

**Technical Features**:
- **Structured Output**: Direct JSON output for compatible providers (Ollama, LlamaCPP) with automatic fallback
- **Model Routing**: Intelligent model selection with provider-specific optimizations
- **Local Model Queuing**: Thread-safe sequential execution for local hardware to prevent resource conflicts
- **Progress Tracking**: Real-time WebSocket updates with detailed agent status and execution metrics
- **Error Handling**: Comprehensive error handling with fallback strategies and graceful degradation

📖 **[Research Agent System Documentation](docs/research_agent_system.md)** - Complete technical documentation with configuration examples and workflow details.

### Profile Management

The profile management system enables researchers to create and manage multiple research profiles, each with its own configuration for email recipients, research interests, ArXiv filters, and visual styling. This system provides a foundation for personalized research workflows across all Theseus Insight features.

#### Core Profile Features

**Multi-Profile Workflows**: Create unlimited research profiles with individual configurations:
- **Email Recipients**: Each profile maintains its own list of newsletter recipients
- **Research Interests**: Profile-specific research interests for targeted content generation
- **ArXiv Category Filters**: Customizable ArXiv category selections with main category and subcategory filtering
- **Visual Styling**: Color-coded profiles with custom descriptions and tags for easy identification
- **Default Profile System**: Designate default profiles for streamlined workflows

**Smart Profile Selection**: System-wide profile selection interface with:
- **Visual Profile Chips**: Individual profile chips with remove buttons for easy management
- **Multi-Profile Support**: Select multiple profiles simultaneously with combined settings
- **Smart Selection Bar**: Consistent profile selection interface across all features
- **Real-Time Stats**: Combined email recipient counts and research interest previews

#### Profile API Endpoints

**`GET /api/profiles`** - List all profiles
- **Returns**: All profiles with metadata, paper counts, and configuration summaries
- **Features**: Includes default profile identification and total paper counts per profile

**`POST /api/profiles`** - Create new profile
- **Parameters**: `name`, `description`, `color`, `tags`, `email_recipients`, `research_interests`, `arxiv_filters`
- **Returns**: Created profile with generated ID
- **Features**: Automatic population from current settings for new profiles

**`GET /api/profiles/{profile_id}`** - Get specific profile details
- **Returns**: Complete profile configuration including research interests and filters

**`PUT /api/profiles/{profile_id}`** - Update profile configuration
- **Parameters**: Any profile field (name, description, color, tags, etc.)
- **Returns**: Updated profile information
- **Features**: Partial updates supported, maintains data integrity

**`DELETE /api/profiles/{profile_id}`** - Delete profile
- **Returns**: Confirmation of deletion
- **Restrictions**: Cannot delete default profiles

**`GET /api/profiles/{profile_id}/interests`** - Get profile research interests
- **Returns**: Detailed research interests with metadata

**`GET /api/profiles/tags`** - Get available tags for autocomplete
- **Returns**: All tags used across profiles for consistent tagging

#### Profile Integration

Profiles integrate seamlessly with all Theseus Insight features:

**Newsletter Generation**: Use profile-specific email recipients and research interests
```bash
POST /api/profiles/{profile_id}/newsletters
```

**Mind-Map Generation**: Profile-seeded mind-maps using profile research interests
```bash
POST /api/profiles/{profile_id}/mindmaps
```

**Research Agent**: Profile-aware research workflows with personalized configurations
```bash
POST /api/profiles/{profile_id}/research
```

**Multi-Profile Operations**: Many endpoints support multiple profile IDs for combined workflows:
- Combined email recipient lists (deduplicated)
- Merged research interests from all selected profiles
- Unified ArXiv filtering across profiles

#### Profile Data Management

**Automatic Settings Population**: New profiles automatically inherit:
- Current email recipients from settings
- Global research interests
- Default ArXiv category filters
- System-wide configuration preferences

**Profile Statistics**: Each profile tracks:
- Total papers associated with the profile
- Email recipient counts
- Research interest complexity
- Usage metrics across features

**Data Integrity**: Profile system maintains:
- Referential integrity with generated content
- Consistent email formatting and validation
- ArXiv category validation against taxonomy
- Tag normalization and deduplication

---

## Database Migration

Theseus Insight includes comprehensive database migration tools for transferring data between environments and migrating from legacy SQLite installations to PostgreSQL.

### SQLite to PostgreSQL Migration

For users upgrading from previous SQLite-based installations:

📖 **[Complete Migration Guide](docs/migration_guide.md)** - Detailed instructions for migrating from SQLite to PostgreSQL.

**Quick Migration Example:**
```bash
# Export from SQLite
python -m theseus_insight.utils.db_migration.db_export \
    --source-db data/theseus.db \
    --output ./sqlite_backup.json

# Import to PostgreSQL
python -m theseus_insight.utils.db_migration.db_import \
    --input ./sqlite_backup.json
```

### PostgreSQL to PostgreSQL Migration

**Export PostgreSQL database to archive:**
```bash
python -m theseus_insight.utils.db_migration.db_export \
    --output ./backup.json
```

**Import archive to new PostgreSQL database:**
```bash
python -m theseus_insight.utils.db_migration.db_import \
    --input ./backup.json
```

**Direct migration with verification:**
```bash
python -m theseus_insight.utils.db_migration.db_migrate \
    --source-db "postgresql://user:pass@old-host:5432/db" \
    --target-db "postgresql://user:pass@new-host:5432/db" \
    --verify
```

### Features
- **Cross-Database Migration** - Full support for SQLite to PostgreSQL migration
- **Intelligent duplicate handling** - Papers detected by URL, podcasts by title, newsletters by date range
- **Vector embedding preservation** - Maintains exact embeddings during migration with pgvector compatibility
- **Schema translation** - Automatic conversion between SQLite and PostgreSQL data types
- **Verification tools** - Compare source and target databases after migration
- **PostgreSQL Optimization** - Optimized imports using PostgreSQL-specific features like COPY and bulk operations

For detailed documentation and advanced usage examples, see [docs/db_migration_README.md](docs/db_migration_README.md).

---

## Credits

- [paperswithcode.com](https://paperswithcode.com/) for research paper data.
- [arxiv.org](https://arxiv.org/) for open access to research paper data.
- [LLMFactory](https://github.com/M-Chimiste/LLMFactory.git) for unified LLM inference across multiple providers.
- [Docling](https://github.com/doclingjs/docling) for document parsing.
- [pydub](https://github.com/jiaaro/pydub) for audio processing.
- [Amazon Polly](https://aws.amazon.com/polly/), [OpenAI TTS](https://platform.openai.com/docs/) for text-to-speech.
- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://pydantic-docs.helpmanual.io/), PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) for backend processing.

Theseus Insight is maintained by [M. Chimiste](https://github.com/M-Chimiste) & contributors.

## Reliability and local verification

See [the reliability guide](docs/reliability.md) for locked installation, the local quality gate, credential migration, authenticated access, job recovery, diagnostics, and search/evaluation commands. No GitHub workflows are required.

See [newsletter quality](docs/newsletter-quality.md) for local-model evidence extraction, editorial selection, source checks, and draft-only evaluation that never sends email.
