# CI/CD Workflows

This directory contains GitHub Actions workflows for automated testing and quality assurance.

## `test.yml` - Test Workflow

### Overview
Lightweight CI workflow designed for efficient use of GitHub Actions free quota.

### Triggers
- Push to `main` or `dev` branches
- Pull requests targeting `main` or `dev` branches

### Jobs
1. **Setup** (Python 3.11 + Ubuntu latest)
2. **Linting** (Fast ruff check with basic rules)
3. **Testing** (Config module tests only)
4. **Coverage** (If tests pass)

### Quota Optimization
- **Single environment**: No matrix builds
- **Pip caching**: Faster dependency installation
- **Focused scope**: Only tests configuration module
- **Smart triggers**: Avoids running on feature branch commits
- **Continue-on-error**: Linting failures don't stop tests

### Local Testing
Before pushing, run tests locally:
```bash
# Run the same tests as CI
pytest tests/test_imports.py tests/config/ -v

# Check linting
ruff check adni_classification tests --select E,W,F --ignore E501

# Generate coverage report
pytest tests/test_imports.py tests/config/ --cov=adni_classification --cov=adni_flwr --cov-report=term-missing
```

### Badge Status
The workflow status is displayed in the README.md with:
```markdown
[![Tests](https://github.com/Tin-Hoang/fl-adni-classification/workflows/Tests/badge.svg)](https://github.com/Tin-Hoang/fl-adni-classification/actions/workflows/test.yml)
```

### Expected Runtime
- **Total**: ~2-3 minutes per run
- **Setup**: ~30 seconds
- **Dependencies**: ~60 seconds (with cache: ~10 seconds)
- **Linting**: ~10 seconds
- **Tests**: ~30 seconds
- **Coverage**: ~20 seconds

This efficient design maximizes value while minimizing GitHub Actions quota usage.
