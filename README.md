# GitHub Devin Integration

A powerful CLI tool to integrate with GitHub repositories and interact with Devin for issue analysis and resolution.

## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install from Source](#install-from-source)
  - [Install from PyPI](#install-from-pypi)
- [Setup & Configuration](#setup--configuration)
  - [Environment Configuration](#environment-configuration)
  - [Security Note](#security-note)
- [Usage](#usage)
  - [Running Directly](#running-directly)
  - [Using Installed Package](#using-installed-package)
- [Commands](#commands)
  - [List Issues](#list-issues)
  - [Analyze an Issue](#analyze-an-issue)
  - [Resolve an Issue](#resolve-an-issue)
  - [Check Session Status](#check-session-status)
  - [Batch Analyze Issues](#batch-analyze-issues)
  - [Interactive Setup](#interactive-setup)
- [Development](#development)
- [License](#license)

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- creating conda environment and installing the `requirements.txt` or use `environment.yml` - This is optional but recommended
- GitHub Personal Access Token with appropriate permissions
- Devin API Key for analysis and resolution of the GitHub issue

### Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/jalajthanaki/github-devin-integration.git
   cd github-devin-integration
   ```

2. Install the package in development mode:
   ```bash
   # Install build tools
   python -m pip install --upgrade build twine
   
   # Build the package
   python -m build
   
   # This will create two files in dist/:
   # - dist/github_devin_integration-0.1.0.tar.gz
   # - dist/github_devin_integration-0.1.0-py3-none-any.whl
   
   # Install the package (choose one method):
   # Option 1: Install the built wheel
   pip install dist/github_devin_integration-0.1.0-py3-none-any.whl
   
   # Option 2: Install in development mode
   pip install -e .
   ```

### Install from PyPI

```bash
# Install from PyPI (testpypi)
pip install -i https://test.pypi.org/simple/ github-devin-integration==0.1.0

# Install from PyPI (pypi) - TODO - Yet to commit on the pypi
pip install github-devin-integration
```

## Setup & Configuration

### Environment Variables

You can configure the tool using either environment variables or a `.env` file.

#### Option 1: Environment Variables
```bash
export GITHUB_TOKEN=your_github_token
export DEVIN_API_KEY=your_devin_api_key  # Required for Devin API
```

#### Option 2: .env File
1. Create a `.env` file in one of these locations (checked in order):
   - `~/.github-devin/.env`
   - `~/.config/github-devin/.env`
   - Current working directory

2. Use the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your credentials:
   ```env
   # GitHub Personal Access Token with 'repo' scope
   GITHUB_TOKEN=your_github_token_here
   
   # Devin API Key (required for analysis features)
   DEVIN_API_KEY=your_devin_api_key_here
   ```

### Interactive Setup

For first-time setup, you can use the interactive wizard:
```bash
github-devin setup
```

### Security

- The `.env` file is excluded from version control by default
- Never commit sensitive credentials to version control
- Keep your API tokens secure and never share them

## Usage

### Running Directly from Source

Run the CLI directly from the source code:

```bash
# Basic syntax
python -m github_devin_integration.cli --repo owner/repo COMMAND [OPTIONS]

# Example:
python -m github_devin_integration.cli --repo octocat/Hello-World list-issues
```

### Using Installed Package

After installation, use the `github-devin` command:

```bash
# Basic syntax
github-devin --repo owner/repo COMMAND [OPTIONS]

# Example:
github-devin --repo octocat/Hello-World list-issues
```

## Commands

### List Issues

List and filter issues in a repository.

#### Basic Usage
```bash
# List open issues (default)
github-devin --repo owner/repo list-issues --state open --limit 100

# List closed issues
github-devin --repo owner/repo list-issues --state closed --limit 100

# List all issues (open and closed)
github-devin --repo owner/repo list-issues --state all --limit 100
```

#### Filtering Options
```bash
# By label
github-devin --repo owner/repo list-issues --label bug --limit 100

# By assignee
github-devin --repo owner/repo list-issues --assignee username --limit 100

```

### Analyze an Issue

Analyze an issue using Devin's AI capabilities.

```bash
# Basic analysis
github-devin --repo owner/repo analyze-issue 42

# Monitor analysis progress
github-devin --repo owner/repo analyze-issue 42 --monitor

# Use existing session
# (Get session ID from previous analysis or session list)
github-devin --repo owner/repo analyze-issue 42 --session-id devin-123 --monitor
```

### Resolve an Issue

Start or continue working on resolving an issue with Devin.

```bash
# Basic resolution
github-devin --repo owner/repo resolve-issue 42

# Monitor resolution progress
github-devin --repo owner/repo resolve-issue 42 --monitor

# With custom timeout (in minutes)
github-devin --repo owner/repo resolve-issue 42 --session-id devin-123 --monitor --max-wait 60
```

### Check Session Status

Check the status of an existing Devin session.

```bash
# Basic status check
github-devin --repo owner/repo session-status devin-123

```

### Batch Analyze Issues

Analyze multiple issues concurrently.
Limitation: Long output is truncated. You cannot see sometimes confidence score.

```bash
# Analyze specific issues
github-devin --repo owner/repo batch-analyze 42 13 7
```

## Development

### Prerequisites
- Python 3.8+
- Poetry (recommended) or pip
- Pre-commit hooks (optional but recommended)

### Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -e .[dev]
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=github_devin_integration tests/
```

### Building the Package

```bash
# Build the package
python -m build

# Check package quality
python -m twine check dist/*
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
