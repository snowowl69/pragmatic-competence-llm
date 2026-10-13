# Contributing

Read the [project documentation](docs/README.md) and current [LICENSE](LICENSE) first. This guide describes the review workflow; it does not change licensing terms or grant permission to redistribute third-party materials.

For a bug or proposed change, open an issue describing the problem, expected behavior and a minimal example. Keep personal information, tokens and private dataset records out of public issues. Use a focused branch and pull request for code changes.

Install core requirements, then validate relevant changes:

```bash
python -m pytest -q
python -m compileall -q src app
```

Run training-specific checks only with the optional training environment. Explain which checks ran, what could not run and why. Include genuine screenshots for interface changes. Do not present mocked tests or CPU results as completed LLM experiments.

Preserve benchmark IDs, split integrity and model/prompt/context identity in evaluations. Document any label-map or data change and regenerate affected artifacts deliberately. Never tune on held-out test outcomes. Add a regression check for substantive behavior fixes; avoid tests that merely repeat implementation details.

Do not commit credentials, personal reviewer identifiers, external licensed model weights or unrelated generated caches. Include attribution and provenance for new data. Pull requests should explain the problem, final behavior, validation and remaining limitations.
