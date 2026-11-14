# Security

## Reporting a vulnerability

Please report it privately, not in a public issue: open the repository's
**Security** tab and choose **Report a vulnerability**. That opens a private
advisory only the maintainer can read.

Say what you found, how to reproduce it, and what it lets someone do.

## What is in scope

- The scoring service (`src/serving/`, the `Dockerfile`): anything that lets a
  caller read another request's data, write applicant data to the logs, change
  a score, or run code.
- The pinned dependencies in `requirements.txt` and `requirements-serve.txt`.

The service is a demonstration and ships without authentication or rate
limiting by design; `GET /model` publishes the full points table on purpose.
Neither is a vulnerability here, and both would have to change before the
service scored real applicants.
