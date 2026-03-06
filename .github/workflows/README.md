# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing and quality assurance.

## `functional_test.yml` - Functional Test Workflow

### Overview
Lightweight CI workflow designed for efficient use of GitHub Actions free quota. Uses `uv` for ultra-fast Python package management.

### Triggers
- Push to `main` or `dev` branches
- Pull requests targeting `main` or `dev` branches

### Jobs
1. **Setup** (Python 3.11 + Ubuntu latest + uv installation)
2. **Dependencies** (uv sync --dev for ultra-fast installation)
3. **Linting** (Fast ruff check with basic rules)
4. **Testing** (Import tests + Config module tests)
5. **Coverage** (If tests pass)

### Quota Optimization
- **Single environment**: No matrix builds
- **UV package manager**: Ultra-fast dependency installation (faster than pip)
- **Focused scope**: Only tests configuration module
- **Smart triggers**: Avoids running on feature branch commits
- **Continue-on-error**: Linting failures don't stop tests

### Local Testing
Before pushing, run tests locally:
```bash
# Run the same tests as CI
uv run pytest tests/test_imports.py tests/config/ -v

# Check linting
uv run ruff check adni_classification tests --select E,W,F --ignore E501

# Generate coverage report
uv run pytest tests/test_imports.py tests/config/ --cov=adni_classification --cov-report=term-missing
```
