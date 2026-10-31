# Presentación de la defensa — ERAE

Láminas para la defensa oral del Trabajo de Grado, hechas con [Slidev](https://sli.dev/).
El contenido sale del informe (`../report/report.typ`); las reglas de contenido, estilo y
tiempos están en [`CLAUDE.md`](./CLAUDE.md), que es la referencia tanto para nosotros como
para Claude Code.

## Requisitos

- Node.js (LTS reciente) y [pnpm](https://pnpm.io/).
- Para exportar a PDF, `playwright-chromium`; `pnpm-workspace.yaml` ya autoriza su
  script de instalación.

## Comandos

```bash
pnpm install       # dependencias (versiones fijadas en package.json)
pnpm run dev       # servidor local en http://localhost:3030 (modo presentador: tecla p)
pnpm run build     # SPA estática en dist/, para presentar sin internet
pnpm run export    # PDF de respaldo (videos e iframes salen como imagen fija)
```

## Estructura

| Ruta | Contenido |
|---|---|
| `slides.md` | Encabezado del deck e importación (`src:`) de cada sección |
| `pages/01-problema.md` … `06-conclusiones.md` | Una sección por archivo, para editar en paralelo sin conflictos |
| `pages/99-respaldo.md` | Láminas de respaldo para las preguntas del jurado, después de «Gracias» |
| `public/` | Imágenes, video de la demo y demás recursos locales |
| `components/` | Componentes Vue propios del deck |
| `OUTLINE.md` | Mapa del deck: tiempos por sección, inconsistencias y preguntas generales |

Las notas de cada lámina son el guion: empiezan con `PRESENTA: <nombre> · TIEMPO: <s>s`, y
las dudas pendientes de esa lámina aparecen como `PREGUNTA:` para resolverlas al revisar.

## Recursos generados (no versionados)

Algunos archivos solo existen en la máquina que los generó; tras un clon hay que
reponerlos antes de `build` o `export`:

- **Visores interactivos de C4** para el respaldo, en `public/c4/`. Se generan con Archify
  (ver [`../docs/architecture/README.md`](../docs/architecture/README.md)) y se copian:

  ```bash
  mkdir -p public/c4 && cp ../docs/architecture/c4-0*.html public/c4/
  ```

- **Video de la demo**, si pesa demasiado para versionarlo: pedírselo a quien lo grabó y
  colocarlo en `public/` con el nombre que referencian las láminas.

Las figuras estáticas (`c4-*.svg`, fotos de prototipos) sí están versionadas en
`../report/images/` y se copian a `public/` con el mismo nombre.