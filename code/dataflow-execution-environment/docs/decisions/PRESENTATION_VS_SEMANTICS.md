# Separación entre semántica CPA y modo de presentación

**Status:** Implementado, ADR formal pendiente.

## Contexto

Durante la Etapa 1 del diseño de ERAE, se identificó la necesidad de distinguir entre
qué ES un valor (su categoría CPA semántica: concreto, pictórico o abstracto) y cómo
se MUESTRA ese valor al usuario. Originalmente ambos conceptos estaban acoplados,
lo que impedía mostrar un valor abstracto con una representación pictórica sin
modificar el programa subyacente.

## Decisión

Se separaron los dos conceptos en capas distintas:
- **Semántica CPA** (`CPAObject.category` en el interpreter): define la naturaleza
  del valor según fue declarado o computado en el programa.
- **Modo de presentación** (`ResultViewMode` en el frontend): define cómo el sandbox
  renderiza visualmente ese valor, controlable por el docente en tiempo real.

## Consecuencias

- El docente puede alternar libremente entre representaciones (pictórico ↔ concreto ↔ abstracto) sin modificar el código del programa.
- Se habilita la técnica de "concreteness fading" en tiempo real durante la clase.
- El intérprete permanece agnóstico al modo de visualización, simplificando su lógica.
- La semántica del programa es determinista e independiente de preferencias de UI.
- Se desacopla la capa de presentación de la capa de ejecución.

## Referencias

- Bruner (1966)
- Vergnaud (1983)
- Haylock & Cockburn (1989)
- Fyfe et al. (2014)

---

> **Nota:** Este archivo es provisional. Se reemplazará por ADR-009 formal cuando
> se priorice documentación arquitectónica.
