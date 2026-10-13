# Troubleshooting

| Problem | Check or action |
|---|---|
| Browser refuses the connection | Start the app, wait for its local URL, and keep the process running. Match the URL to the selected port. |
| Port is already occupied | Use another port, such as `--port 7866`, or stop the old project process you started. |
| CPU classifier file is missing | Run `python src/evaluation/local_baseline.py` after installing core requirements and preparing v2 data. |
| Dependency mismatch when loading the CPU model | Use a clean environment; if necessary rebuild the CPU reference in that environment and preserve the old results. |
| Llama access denied | Confirm authorized model access and Hugging Face authentication in the runtime. Do not put a token in a GitHub issue. |
| CUDA unavailable or insufficient memory | Use an adequate CUDA GPU runtime. CPU mode is available for the reference demo but does not replace the 8B experiment. |
| Missing training plot | A configuration is not telemetry. Confirm the training process writes `live_status.json` and `training_metrics.json` beside the dashboard. |
| Training state is stale | Inspect the actual training process and logs. A slow step and a stopped process can both cause an old timestamp. |
| Empty review queue | Generate the separate interpretation sample; label-only classifications are not review samples. Confirm benchmark hashes match. |
| Missing transfer cells | Complete the relevant evaluation condition with matching test coverage and settings. Missing is not zero. |
| Tables extend beyond the screen | Scroll horizontally or use the table's fullscreen control. |

For a bug report, include the command, environment versions, a minimal non-sensitive example, expected/actual behavior and a sanitized traceback. Use the repository's issue template. For security-sensitive reports, follow [security guidance](../SECURITY.md).
