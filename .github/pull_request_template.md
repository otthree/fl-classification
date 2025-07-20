# Pull Request

## Description
Brief description of the changes in this PR.

Fixes #(issue_number) <!-- Remove if not applicable -->

## Type of Change
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🧪 Research/experimental change
- [ ] 🔧 Configuration change
- [ ] ♻️ Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security improvement

## Federated Learning Impact
- [ ] Affects FedAvg strategy
- [ ] Affects FedProx strategy
- [ ] Affects SecAgg/SecAggPlus strategies
- [ ] Introduces new FL strategy
- [ ] Changes client-side logic
- [ ] Changes server-side logic
- [ ] Affects aggregation algorithms
- [ ] No FL impact

## Component Changes
- [ ] Model architectures (`adni_classification/models/`)
- [ ] Dataset handling (`adni_classification/datasets/`)
- [ ] FL implementation (`adni_flwr/`)
- [ ] Training scripts (`scripts/`)
- [ ] Configuration system (`configs/`)
- [ ] Utilities (`adni_classification/utils/`)
- [ ] Documentation
- [ ] Tests
- [ ] Dependencies

## Testing
### Test Coverage
- [ ] Added new tests for new functionality
- [ ] Updated existing tests
- [ ] All tests pass locally
- [ ] Manual testing completed

### FL Testing
- [ ] Tested with local simulation (`run_local_simulation.py`)
- [ ] Tested with multi-machine setup (`run_multi_machines_tmux.py`)
- [ ] Tested with different client counts (2, 3, 4)
- [ ] Tested with different model architectures
- [ ] Verified privacy preservation (if applicable)

### Research Validation
- [ ] Verified results against baseline
- [ ] Compared with centralized training
- [ ] Statistical significance testing (if applicable)
- [ ] Performance metrics documented

## Configuration Impact
**Breaking changes to configuration?** Yes/No

If yes, describe:
- Which config files are affected
- Migration path for existing configs
- Backward compatibility considerations

## Performance Impact
- [ ] No performance impact
- [ ] Improves performance
- [ ] May decrease performance
- [ ] Performance impact unknown

**Benchmarks** (if applicable):
- Training time: Before/After
- Memory usage: Before/After
- Model accuracy: Before/After
- Communication overhead: Before/After

## Research Context
**Research Motivation**: Why is this change scientifically relevant?

**Methodology**: How does this change affect the experimental methodology?

**Expected Outcomes**: What research outcomes does this enable or improve?

## Checklist
### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of the code completed
- [ ] Code is properly commented/documented
- [ ] No obvious performance issues
- [ ] No security vulnerabilities introduced

### Documentation
- [ ] Updated relevant documentation
- [ ] Added/updated docstrings
- [ ] Updated README if necessary
- [ ] Added usage examples (if applicable)

### Dependencies
- [ ] No new dependencies added
- [ ] New dependencies justified and documented
- [ ] Updated `pyproject.toml`/`requirements.txt` if needed
- [ ] Dependencies are compatible with existing versions

### Research Reproducibility
- [ ] Changes maintain reproducibility
- [ ] Random seeds handled appropriately
- [ ] Configuration files updated
- [ ] Results can be replicated

## Screenshots/Results
If applicable, add screenshots, plots, or experimental results.

## Additional Notes
- Any additional context about the changes
- Future work considerations
- Known limitations or issues
- Reviewers to specifically tag
