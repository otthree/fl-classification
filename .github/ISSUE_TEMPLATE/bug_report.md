---
name: Bug Report
about: Create a report to help us improve the federated learning implementation
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Environment
- **OS**: [e.g., Ubuntu 20.04, macOS 12.0]
- **Python Version**: [e.g., 3.11.0]
- **UV Version**: [run `uv --version`]
- **CUDA Version**: [if applicable, run `nvidia-smi`]
- **GPU Model**: [if applicable]

## Federated Learning Context
- **FL Strategy**: [e.g., FedAvg, FedProx, DP, SecAggPlus]
- **Number of Clients**: [e.g., 2, 3, 4, or centralized]
- **Model Architecture**: [e.g., ResNet3D, DenseNet3D]
- **Configuration File**: [path to config file used]

## Steps to Reproduce
1. Go to '...'
2. Run command '...'
3. Set configuration '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Error Output
```
# Paste the complete error message/traceback here
```

## Log Files
Please attach relevant log files from the `logs/` directory or paste key excerpts:
```
# Paste relevant log output here
```

## Configuration
Please share the configuration file or relevant sections:
```yaml
# Paste your configuration here
```

## Additional Context
- Are you running local simulation or distributed training?
- Any custom modifications to the codebase?
- Dataset size and distribution?
- Any other context that might be helpful

## Screenshots
If applicable, add screenshots to help explain your problem.

## Checklist
- [ ] I have searched existing issues to ensure this is not a duplicate
- [ ] I have included all relevant environment information
- [ ] I have provided a minimal reproducible example
- [ ] I have checked the documentation and README
- [ ] I have included relevant log files or error messages
