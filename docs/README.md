# omi_physics documentation

Deep-dive documentation for the `omi_physics` engine. Start with the
[project README](../README.md) for install and quick start.

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the layers, every module's job, and the
  design principles (structure-of-arrays, renderer-agnostic core, optional
  accelerators).
- **[PIPELINE.md](PIPELINE.md)** — what one `world.step(dt)` does, stage by stage,
  and the data that flows between stages.
- **[DATA-MODEL.md](DATA-MODEL.md)** — the OMI glTF data model, the
  structure-of-arrays world state, and glTF round-tripping.
- **[ACCELERATORS.md](ACCELERATORS.md)** — the Cython accelerators, the
  pure-Python fallback contract, and how they are kept honest.

> The diagrams are [Mermaid](https://mermaid.js.org/); GitHub renders them inline.
