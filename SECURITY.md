# Security Policy

## Reporting a vulnerability

Report security problems privately. Do not open a public issue for a suspected
data leak.

- Use GitHub's private vulnerability reporting for this repository.
- If it is unavailable, use a private contact channel published by the project
  owner rather than posting document data in a public issue.

Include the version or commit, command and relevant flags, expected and actual
behavior, and a **synthetic** reproducer. Never attach real PII, confidential
documents, credentials, or unredacted reports. For a layout-specific issue,
recreate only the layout and invented value shape needed to show the problem.

We aim to acknowledge reports within seven days and provide a fix or plan within
30 days. Credit is optional and will be given only with the reporter's consent.

## What is in scope

Examples include:

- an identifier that remains readable in the rendered output;
- an identifier left in output metadata, annotations, form fields, embedded
  attachments, or another document layer;
- a log or default report that exposes an original value;
- data leaving the machine without the user selecting a remote integration;
- a path traversal, unsafe temporary-file, or dependency issue that can expose
  document contents.

## Limits and safe operation

`pii-mask-lito` is a best-effort masking tool, not a compliance guarantee. It
does not by itself establish de-identification under HIPAA, GDPR, or any other
law, standard, contract, or security program.

Detection can fail because of unusual layouts, unsupported languages, weak OCR,
handwriting, small or rotated text, visual content, embedded data, or identifier
types outside the active policy. A successful verification result only means
the tool's configured checks found no remaining issue; it is not proof that no
identifier remains.

Review rendered output before it leaves your control. Use a policy selected and
validated for the document class and risk level. Keep source documents, masked
outputs, and reports in appropriate access-controlled storage.

By default, core processing is local. Optional hosted `--agents` integrations
can transmit page images and OCR-derived content to a selected provider. Enable
them only after approving that provider and transfer for the data involved.

`--report-values` writes original detected values for an audit workflow. It
turns the report into sensitive data and should be protected like the input.

## Project data handling

Never add real documents, identifiers, screenshots, secrets, or masking reports
to this repository. Keep generated artifacts in ignored directories, review
`git status` before a commit, and use synthetic reproductions in issues and pull
requests.
