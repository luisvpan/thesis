# Diagramas de arquitectura (modelo C4)

Seis diagramas del ambiente ERAE, siguiendo el [modelo C4](https://c4model.com/) y generados con la
herramienta Archify a partir del código en la revisión `a04b8fc`. Cada uno existe en dos formas:

- **HTML explorable** (`c4-0N-*.html`): página autocontenida con temas claro y oscuro, búsqueda de
  nodos, trazado de rutas y enlaces a la evidencia en el repositorio. Se abre en cualquier navegador.
- **SVG** (`../../report/images/c4-*.svg`): la misma imagen, con los estilos CSS ya resueltos para que
  Typst la componga en el informe.

| Diagrama | Nivel C4 | Dónde aparece en el informe |
|---|---|---|
| `c4-01-paisaje` | Paisaje del sistema | Cap. IV, *Arquitectura física y lógica del ambiente* |
| `c4-02-contexto` | Nivel 1 — Contexto | Apéndice B §3.1 |
| `c4-03-contenedores` | Nivel 2 — Contenedores | Apéndice B §3.1 |
| `c4-04-componentes-vision` | Nivel 3 — Componentes | Apéndice B §3.2 |
| `c4-05-componentes-interfaz` | Nivel 3 — Componentes | Apéndice B §3.2 |
| `c4-06-secuencia-ciclo` | Dinámico — Secuencia | Apéndice B §3.3 |

Las tarjetas laterales (notas con la evidencia detallada) solo aparecen en el HTML; el SVG del
informe contiene únicamente el dibujo.

## Cómo regenerar un diagrama

Las fuentes editables están en `sources/`. Cada archivo es el candidato de Archify **sin** el bloque
`meta.translations`, que se inyecta al construir.

```bash
# desde la raíz del repositorio
node <ruta-a-archify>/bin/archify.mjs finalize <tipo> <candidato.json> <salida.html> \
  --repo-root . --quality showcase --json
```

donde `<tipo>` es `architecture` para los cinco primeros y `sequence` para el último. El indicador
`--repo-root .` es obligatorio: los diagramas declaran evidencia (`sources`) y Archify la verifica
contra los bytes confirmados en la revisión fijada en `meta.repository` antes de dibujar.

## Cómo reexportar el SVG para el informe

El visor de Archify exporta SVG desde su menú *Exportar*, pero ese SVG usa propiedades personalizadas
de CSS (`var(--…)`) que **resvg, el motor SVG de Typst, no implementa**: sin resolverlas, todas las
figuras salen en negro. El SVG guardado en `report/images/` tiene esas variables ya sustituidas por
el valor que calcula el navegador con el tema claro. Si se regenera un diagrama hay que repetir esa
sustitución, no basta con descargar el SVG del visor.

`tools/export-svg.mjs` hace las dos cosas en un paso: abre el HTML en el mismo Chrome sin cabeza que
usa Archify, dispara la exportación en tema claro y sustituye las variables antes de guardar.

```bash
# desde la raíz del repositorio
node docs/architecture/tools/export-svg.mjs \
  docs/architecture/c4-03-contenedores.html report/images/c4-contenedores.svg
```

Busca el paquete de Archify en `~/.claude/skills/archify`; para otra ubicación, indicar
`ARCHIFY_HOME`.

## Nota sobre el tamaño del texto

El ancho útil del informe es de 165 mm (US Letter con márgenes de 2,54 cm). Los diagramas se
dimensionaron para que la etiqueta principal de cada nodo quede entre 6,5 y 9 pt impresa; por eso son
estrechos y altos en lugar de anchos. El diagrama de secuencia, que es ancho por naturaleza, ocupa una
página apaisada.
