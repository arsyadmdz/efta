# Changelog

All notable changes to efta are documented here.

## [1.0.0] — 2026-04-06

### Added
- Initial PyPI release.
- `reaction` and `reactions` classes for defining and solving coupled chemical equilibria.
- Four numerical solver methods: Method L (Newton in log-space), Method A (extent-of-reaction), Method B (fsolve variants), Method DE (differential evolution).
- `solution` and `mixture` classes for managing equilibrium results.
- Solvent extraction module (`efta.solventextraction`) with single-stage `sx` and multistage topologies (`countercurrent`, `crosscurrent`, `strip_countercurrent`, `strip_crosscurrent`).
- Activity coefficient support via `reaction.set_gamma()` and `reactions.set_gamma()`.
- Precipitation/Ksp reactions (`ksp=True`).
- Parameter fitting (`model`, `analyze`, `montecarlo`) with bootstrap uncertainty.
- `freaction` / `freactions` for parameterised reactions with `$(xN)` placeholders.
- Full periodic table (`periodic_table`) with IUPAC 2021 atomic masses.
- Plotting utilities: concentration sweeps, speciation fraction diagrams, stage profiles.
- `PlotStyle` singleton (`efta.style`) for consistent plot customisation.
- Styling helpers: `coloring()`, `randomize_color()`, built-in palettes including `'colorblind'`, `'ocean'`, `'earth'`.
- Complete error hierarchy (`EftaError`, `ConvergenceError`, `ConvergenceWarning`, …).
- `py.typed` marker — efta is fully typed.
