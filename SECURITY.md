# Security

This is a research application. It has no documented production security audit or guaranteed support window. Reports should identify the affected commit and environment.

## Reporting

Do not post access tokens, credentials, private records or actionable sensitive vulnerability details in public issues. If GitHub offers private vulnerability reporting for this repository, use that channel. Otherwise contact the repository owner through a contact method they publish, or open a minimal issue requesting a private contact channel without disclosing the vulnerability. No private reporting service or response-time commitment is implied here.

Include the affected component, reproduction steps, impact and a suggested mitigation where possible. Use synthetic examples instead of personal data.

## Operational precautions

- Run the demo locally by default. Public sharing changes who can access the application; the human-review page is not an authenticated reviewer portal.
- Load `.joblib` and other serialized Python objects only from trusted sources. Untrusted serialization may execute code.
- Keep Hugging Face and other service tokens outside tracked files and screenshots.
- Treat contexts, model outputs and reviewer notes as potentially sensitive. Review data before publishing artifacts.
- Check archive paths before extracting external result bundles, and retain backups before replacing experimental evidence.

Security fixes should preserve the distinction between real research evidence and synthetic tests.
