# Building the docs

The documentation is a [MkDocs](https://www.mkdocs.org/) site using the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. API pages are
generated from the docstrings in `src/gfnx` by
[mkdocstrings](https://mkdocstrings.github.io/). Everything lives in
`gfnx-docs/`:

```text
gfnx-docs/
├── mkdocs.yml          # site configuration and navigation
├── requirements.txt    # documentation toolchain
└── docs/               # Markdown sources (this page is docs/building_docs.md)
```

## Build locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r gfnx-docs/requirements.txt
```

You do not need to install `gfnx` itself: mkdocstrings reads the docstrings
statically from `src/` (see `paths: [../src]` in `mkdocs.yml`).

Serve the site with live reload while you edit:

```bash
mkdocs serve -f gfnx-docs/mkdocs.yml
```

Then open <http://127.0.0.1:8000/en/latest/>. The `/en/latest/` prefix comes
from `site_url` in `mkdocs.yml`, which matches the Read the Docs layout.
Changes to Markdown files *and* to docstrings in `src/gfnx` are picked up
automatically.

Build the static site and fail on any warning (broken links, unknown
cross-references, docstring parameters that do not match the signature):

```bash
mkdocs build -f gfnx-docs/mkdocs.yml --strict -d site
```

The output is written to `site/` (ignored by git); open `site/index.html` or
serve it with `python -m http.server -d site`.

## Writing documentation

- **Environments and reward modules are documented in their docstrings.** The
  family pages under `environments/` embed the class docstrings with
  `::: gfnx.environment.<module>.<Class>` blocks, so edit the docstring rather
  than copying the specification into Markdown.
- Docstrings use the Google style (`Args:`, `Returns:`, `Attributes:`,
  `References:`). Use raw strings (`r"""..."""`) when they contain LaTeX.
- Math uses `$...$` inline and `$$...$$` for display, rendered by MathJax.
- Cross-reference API objects with
  ``[`HypergridEnvironment`][gfnx.environment.hypergrid.HypergridEnvironment]``.
- New pages must be added to the `nav` section of `mkdocs.yml`.

## Read the Docs

The hosted site at <https://gfnx.readthedocs.io/> is built by Read the Docs
from `.readthedocs.yaml` at the repository root, which installs
`gfnx-docs/requirements.txt` and runs MkDocs with `gfnx-docs/mkdocs.yml`.
