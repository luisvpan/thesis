<script setup lang="ts">
// Per-slide layer: eyebrow (from the `eyebrow` frontmatter key) and the slide-number footer.
// Main deck: "NN / total", where total counts up to the slide marked `deckEnd: true` ("Gracias").
// Backup slides (after that one): "R-NN". The title slide (no. 1) shows no footer.
import { computed } from 'vue'
import { useNav, useSlideContext } from '@slidev/client'

const { $page, $frontmatter } = useSlideContext()
const { slides } = useNav()

const mainTotal = computed(() => {
  const end = slides.value.findIndex(s => s.meta?.slide?.frontmatter?.deckEnd)
  return end === -1 ? slides.value.length : end + 1
})

const pad = (n: number) => String(n).padStart(2, '0')

const label = computed(() => {
  if ($page.value === 1)
    return ''
  if ($page.value > mainTotal.value)
    return `R-${pad($page.value - mainTotal.value)}`
  return `${pad($page.value)} / ${pad(mainTotal.value)}`
})
</script>

<template>
  <div v-if="$frontmatter.eyebrow" class="eyebrow">
    {{ $frontmatter.eyebrow }}
  </div>
  <div v-if="label" class="slide-number">
    {{ label }}
  </div>
</template>

<style scoped>
.eyebrow {
  position: absolute;
  top: 18px;
  left: 3.5rem;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: #4b5563;
  z-index: 10;
}
.slide-number {
  position: absolute;
  right: 1.25rem;
  bottom: 0.75rem;
  font-size: 16px;
  font-variant-numeric: tabular-nums;
  color: #374151;
  z-index: 10;
}
</style>
