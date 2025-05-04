# MakeHub Metrics Analysis

A Python-based tool for analyzing and verifying MakeHub metrics across different Azure regions.

## Features

- Automated testing of MakeHub endpoints across multiple Azure regions
- Latency and performance analysis
- Token usage tracking and analysis
- Comprehensive logging and error handling
- Data export to CSV format

## Prerequisites

- Python 3.10-3.11
- Poetry package manager
- MakeHub API access
- Azure OpenAI API access in multiple regions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChadyMoukel/makehub-metrics.git
cd makehub-metrics
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Create a `.env` file with your API credentials:
```bash
# MakeHub credentials
MAKEHUB_API_KEY=your_makehub_api_key_here
MAKEHUB_API_URL=https://api.makehub.ai  # Optional, defaults to this value

# Azure OpenAI credentials for each region
AOAI_STUDIO_EAST_US_ENDPOINT=your_eastus_endpoint
AOAI_STUDIO_EAST_US_API_KEY=your_eastus_key

AOAI_STUDIO_GERMANY_WEST_CENTRAL_ENDPOINT=your_germany_endpoint
AOAI_STUDIO_GERMANY_WEST_CENTRAL_API_KEY=your_germany_key

AOAI_STUDIO_NORWAY_EAST_ENDPOINT=your_norway_endpoint
AOAI_STUDIO_NORWAY_EAST_API_KEY=your_norway_key

AOAI_STUDIO_SWEDEN_CENTRAL_ENDPOINT=your_sweden_endpoint
AOAI_STUDIO_SWEDEN_CENTRAL_API_KEY=your_sweden_key

AOAI_STUDIO_SWITZERLAND_NORTH_ENDPOINT=your_switzerland_endpoint
AOAI_STUDIO_SWITZERLAND_NORTH_API_KEY=your_switzerland_key

AOAI_STUDIO_UK_SOUTH_ENDPOINT=your_uk_endpoint
AOAI_STUDIO_UK_SOUTH_API_KEY=your_uk_key

AOAI_STUDIO_FRANCE_CENTRAL_ENDPOINT=your_france_endpoint
AOAI_STUDIO_FRANCE_CENTRAL_API_KEY=your_france_key
```

## Usage

### Running Tests

To run the MakeHub verification tests:

```bash
# Run with default settings
poetry run makehub-verify

# Run with custom settings
poetry run makehub-verify --tests 10 --windows 2 4 8 --global-time 30
```

Available options:
- `--tests`: Number of test iterations to run (default: 5)
- `--windows`: MakeHub metrics windows in minutes (default: [2, 4, 8])
- `--output`: Output CSV file path (default: auto-generated)
- `--global-time`: Global time between iteration starts in seconds (default: 60.0)

### Analyzing Data

To analyze the collected metrics:

```bash
poetry run makehub-analyze
```

### Token Counting

To analyze token usage:

```bash
poetry run makehub-tokens
```

## Project Structure

- `src/makehub_metrics/verification.py`: Main script for running MakeHub tests
- `src/makehub_metrics/analysis.py`: Script for analyzing collected metrics
- `src/makehub_metrics/token_count.py`: Script for token usage analysis
- `pyproject.toml`: Project configuration and dependencies
- `.env`: Configuration file for API credentials (not tracked in git)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
