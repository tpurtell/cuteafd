# Usage history and console access

Request accounting is off by default (`--usage on` enables it) until its
serving-path overhead is measured. When on, coordinators keep usage metadata in
`~/.cache/cuteafd/<instance>/usage/usage.sqlite` on the launch host (7 days,
256 MiB by default). Metadata never contains prompts or completions. The
separate full-log tier defaults to 24 hours and 1 GiB in `usage-log.sqlite`.
**With the full log on, user prompts and model outputs are stored in plain
text for the retention period.** Media are replaced by MIME/size/SHA-256
references; credentials and headers are not logged. Serving/gateway payload
capture hooks are a separate rollout; storage alone does not capture bodies.

The launcher prints the console unlock link with its token only to an interactive
terminal (or with `CUTEAFD_PRINT_CONSOLE_LINK=1`); redirected logs get the secret's path instead. The persistent
secret lives at `~/.cache/cuteafd/console/secret` (0600 in an owned 0700
directory). Console cookies do not authorize API calls, and an API key does
not unlock console data. Rotate with `scripts/launch/console-secret.sh rotate`
or `run.sh --rotate-console-secret`; running coordinators reload within 10
seconds. New flags are omitted for older images and older WIP binaries.

Cookie-protected `/console/usage/settings` controls retention, caps, full-log
recording, client IP (off by default), and benchmark recording. Clear the full
log with `POST /console/usage/log/clear`, or both tiers with
`POST /console/usage/clear`. With the coordinator stopped, deleting
`~/.cache/cuteafd/<instance>/usage/usage-log.sqlite*` clears only payloads.
`--usage off` removes request accounting entirely; absent `--usage-dir`, tests
and standalone APIs use an in-memory store. Only daily aggregates may outlive
metadata retention (90 days by default), without request or session IDs.
