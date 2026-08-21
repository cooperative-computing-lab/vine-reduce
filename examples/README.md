# Examples

`examples/quick_start` in this directory is the only example that stays
here — plain Python values chunked out of binary files, no Coffea/awkward
machinery, the fastest way to see `chunk_to_args`, `processors`, and
`reducer` wired together.

The real physics examples live in
[`vine-cms-analysis-stack`](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack),
not here, so there's one canonical copy that stays in sync with that
stack's Coffea/TaskVine setup instead of drifting from it:

- [`examples/cortado`](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack/tree/main/examples/cortado) —
  a HEP skim over synthetic NanoAOD-like ROOT files using
  `VineReduceCoffea`, no real CMS data or `xrootd` access needed.
- [`examples/ttBar`](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack/tree/main/examples/ttBar) —
  the `ttbarEFT` production integration: how a full physics analysis wires
  up channels, histogram selection, and X509 proxy handling around
  `vine_reduce`.

See that repo's [`examples/README.md`](https://github.com/cooperative-computing-lab/vine-cms-analysis-stack/blob/main/examples/README.md)
for a full index, and its top-level README for installation and the
distributor/executor concepts both examples build on.
