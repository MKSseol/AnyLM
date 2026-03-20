# notebooks/ — Jupyter Notebooks

## Role
Used for exploratory analysis, result visualization, and rapid prototyping.
Should not become production code; validated code should be moved to `src/`.

## Naming Convention
`{date}_{topic}.ipynb`
Example: `20250321_nllb_embedding_tsne.ipynb`

## Rules
- Notebooks are "lab notebooks": hypothesis → execution → results → conclusion structure
- Add markdown cell at the top of each cell explaining "what this cell does"
- Refactor useful code discovered in notebooks into `src/`
- Notebook outputs are not included in git (use nbstripout)
- Use `%load_ext autoreload` + `%autoreload 2` in notebooks for instant `src/` change reflection
