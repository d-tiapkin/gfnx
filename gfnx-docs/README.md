# gfnx documentation sources

This folder holds the sources of the documentation hosted at
<https://gfnx.readthedocs.io/>.

- `docs/` — Markdown pages. Start from [`docs/index.md`](docs/index.md).
- `mkdocs.yml` — site configuration and navigation.
- `requirements.txt` — documentation toolchain.

To build and preview the site locally, from the repository root:

```bash
pip install -r gfnx-docs/requirements.txt
mkdocs serve -f gfnx-docs/mkdocs.yml   # http://127.0.0.1:8000/en/latest/
```

See [`docs/building_docs.md`](docs/building_docs.md) for details and writing
conventions.
