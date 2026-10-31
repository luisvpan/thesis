# Slides (presentation/)

Thesis defense deck for ERAE, built with Slidev. Audience: the jury (tutor + two
jurors). Slide text is Spanish; this file and code comments are English.

## Precedence
- The `slidev` skill (`.agents/skills/slidev/`) is the reference for Slidev
  syntax and features: how to do something.
- This file decides what goes on the slides and how it looks: content, structure,
  density, diagrams, media. When the skill suggests something this file rules
  out (e.g. Mermaid/PlantUML diagrams, remote images, dense code walkthroughs),
  this file wins.

## Source of truth
- Content comes from `../report/report.typ` (and `../report/references.bib`,
  `../report/images/`). Never invent results, numbers, quotes or citations; if
  something is not in the report, ask.
- The title and general objective are approved by the school: copy them verbatim.
- If the report and the code disagree, flag it instead of picking one.
- Terminology: the system has an *intérprete* (never compilador/runtime); it is
  single-surface spatial AR (projection and tangibles share one surface).
- Citations on slides: `(Autor, año)` APA style, and every one must exist in
  `references.bib`.

## What the jury scores (Planilla de Evaluación TEG)
- B, shared (20): time 30 min (4), content and organization (4),
  **product demonstration (12)**. The demo is the single biggest item.
- C, per student (20 each): command of the topic (6), delivery (6), answers (8).
  Both students present; every section has an owner (see budget).
- The jury looks for: problem/objectives/importance/limitations, justified
  methodology, results tied to each specific objective, conclusions that
  correlate with objectives and results, ODS goal.

## Structure and time budget (hard limit 30 min; target 28)
| # | Section | Min | Presenter |
|---|---------|-----|-----------|
| 01 | El problema: planteamiento, ODS + meta, objetivos, alcance/limitaciones | 3 | TBD |
| 02 | Marco teórico: only what the results need | 2 | TBD |
| 03 | Marco metodológico: type, design, instruments, development methodology + why | 2 | TBD |
| 04 | Desarrollo y resultados: one block per specific objective (OE1 to OE5) | 11 | TBD |
| 05 | Demostración: recorded video, the live system on standby | 7 | TBD |
| 06 | Conclusiones (one per OE + general) y recomendaciones | 3 | TBD |
| -- | Respaldo (after "Gracias"): for the Q&A | 0 | - |

- ~90 s per content slide is the default; size each section from its minutes.
- Section 04 shows the result (artifact, figure, finding) of each OE, not the
  activity log. ODS appears in 01, 04 and 06.
- Backup slides anticipate jury questions: traceability matrix, expert-judgment
  instruments and verdicts, methodology/tool comparisons, CV pipeline / touch
  classifier detail, limitations. Backup slides are exempt from density rules.

## Per-slide rules (adapted from cc-slidev's checklist)
Critical (fix before presenting):
1. One idea per slide.
2. Title is an assertion (subject + verb + finding), not a label.
   Above it, a small eyebrow with the location: `04 · Desarrollo y resultados · OE3`.
   - Bad: `Validación`  Good: `Ambos expertos aprobaron el ambiente con observaciones menores`
3. ≤6 elements (each bullet, image, diagram, table, code block counts as one).
   Progressive reveal with `v-click` is fine.
4. <50 words of body text (title excluded). Phrases of 3–6 words, no periods.
High:
5. At least one visual per slide (diagram, photo, screenshot, table, code),
   except title, dividers, quotes, objective statements.
6. Type at the default canvas width (980): body ≥18px, headings ≥24px.
   Prefer body ≥20px; classroom projectors wash out small text.
7. Contrast ≥4.5:1; light background with dark text (projector-safe).
8. Meaning never by color alone: add labels, icons or shapes.
Polish:
9. Title + visual alone convey the point (5-second test).
10. ≥10% margins; never shrink text to make content fit: split the slide.
11. Explainable in ~90 s.

## Presenter notes
- Every content slide ends with an HTML comment holding the notes (Slidev
  presenter mode). Slide = what; notes = why and how, in full sentences: the
  spoken script, with the numbers and detail that were cut from the slide.
- First line of the notes: `PRESENTA: <name> · TIEMPO: <seconds>s`.
- Open questions about a specific slide go in its notes, right after the first
  line, one per line as `PREGUNTA: …`, so we read them while reviewing. When we
  answer one, apply it and delete that line. Deck-wide questions go in
  `OUTLINE.md` instead.
- Keep the sum of TIEMPO per section within the budget table; report the
  total whenever slides are added or removed.

## Slidev conventions
- `slides.md` holds only the headmatter and one `src:` import per section:
  `pages/01-problema.md` … `pages/06-conclusiones.md`, `pages/99-respaldo.md`.
  One file per section so both of us can edit in parallel.
- Versions in `package.json` stay pinned exactly; no `^`.
- Everything must work offline: media lives in `public/`, no remote images
  (e.g. the starter's `background: https://cover.sli.dev`), no CDN fonts
  unless bundled. Demo video is a local `.mp4` (`<video controls>` with a
  poster frame), never YouTube/streamed.
- Reuse report figures by copying them from `../report/images/`; keep names.
- Every slide except the title shows a footer with `NN / total` (zero-padded,
  total = main deck only, backup slides excluded), so the jury can cite a slide
  by number in the Q&A. Implement it once (global layer or layout), not per slide.
  Backup slides show `R-NN` instead.

## Diagrams
- Architecture: reuse the Archify C4 SVGs from `../report/images/c4-*.svg`
  (same figures the jury read). Never redraw architecture by hand.
  Main deck: `c4-paisaje` only. If a slide needs a lighter view, make a
  reduced Archify candidate in `../docs/architecture/sources/` and export it
  with `../docs/architecture/tools/export-svg.mjs`.
- Backup: the interactive `c4-0N-*.html` viewers may be embedded in an
  `<iframe>` for Q&A. They are git-ignored build products of
  `archify finalize` (see `../docs/architecture/README.md`), so they are
  copied into `public/c4/` (also git-ignored) to be served by Slidev.
- Non-architecture flows (problem chain, methodology phases): HTML/CSS boxes
  in the slide, or d2 exported to SVG if it already exists in `../wip/`.
- No Mermaid, no PlantUML.

## Generated, git-ignored assets: remind us
These exist only on the machine that generated them, so a fresh clone or the
other student's laptop will not have them:
- `public/c4/c4-0N-*.html` (Archify viewers for the backup slides)
- the demo video in `public/`, if it is too large to commit

At the start of any session that touches the backup section, and before every
`build`/`export`, check that each file referenced from the slides exists. If
any is missing, say so up front, list the missing files and give the command
to regenerate or copy them; don't silently drop or replace the slide.

## Before the defense (surface this list when asked to build or export)
- [ ] Generated assets above present; `pnpm run build` passes offline
      (network disabled) and `dist/` opens on the presenting laptop.
- [ ] `pnpm run export` PDF fallback made (it shows iframes and videos as
      static frames; check those slides still make sense).
- [ ] Timed rehearsal ≤28 min; TIEMPO notes match the budget.
- [ ] Live system (Kinect + projector) set up and calibrated in case the jury
      asks for it.
- [ ] Projector check in the room, at least an hour before.

## Don't
- Don't add slides beyond the budget without saying which section loses time.
- Don't paste report paragraphs onto slides; summarize into phrases and move
  the prose to notes.
- Don't introduce concepts the deck hasn't set up yet.
