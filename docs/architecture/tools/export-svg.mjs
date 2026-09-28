// Exporta a SVG (tema claro) un artefacto HTML de Archify, usando el mismo
// Chrome sin cabeza que emplea `archify browser-check`.
//
// Resuelve las propiedades personalizadas de CSS (`var(--…)`) que el visor deja
// sin expandir, porque resvg —el motor SVG de Typst— no las implementa y sin
// ellas el dibujo sale completamente en negro.
//
//   ARCHIFY_HOME=<ruta-al-paquete-archify> \
//     node export-svg.mjs <entrada.html> <salida.svg> [<entrada.html> <salida.svg> ...]

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const archifyHome =
  process.env.ARCHIFY_HOME || `${process.env.USERPROFILE || process.env.HOME}/.claude/skills/archify`;
const { ChromeVisualBrowser, findChrome } = await import(
  pathToFileURL(path.join(archifyHome, "bin/visual-check.mjs")).href
);

const pairs = [];
for (let i = 2; i < process.argv.length; i += 2) {
  pairs.push([path.resolve(process.argv[i]), path.resolve(process.argv[i + 1])]);
}
if (pairs.length === 0) {
  console.error("uso: node export-svg.mjs <entrada.html> <salida.svg> [...]");
  process.exit(1);
}

const chrome = findChrome();
if (!chrome) {
  console.error("No se encontró Chrome.");
  process.exit(1);
}

const browser = new ChromeVisualBrowser(chrome);
const sessionId = await browser.sessionPromise;
const send = (method, params = {}) => browser.cdp.send(method, params, sessionId);

async function run(expression, awaitPromise = true) {
  const result = await send("Runtime.evaluate", {
    expression,
    awaitPromise,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(
      result.exceptionDetails.exception?.description ||
        result.exceptionDetails.text,
    );
  }
  return result.result?.value;
}

try {
  await browser.cdp.send("Browser.setDownloadBehavior", { behavior: "deny" });
  await send("Emulation.setDeviceMetricsOverride", {
    width: 1600,
    height: 1200,
    deviceScaleFactor: 1,
    mobile: false,
  });

  for (const [htmlPath, svgPath] of pairs) {
    const url = pathToFileURL(htmlPath);
    url.searchParams.set("theme", "light");

    const loaded = browser.cdp.waitFor("Page.loadEventFired", sessionId);
    loaded.catch(() => {});
    const navigation = await send("Page.navigate", { url: url.href });
    if (navigation.errorText) {
      throw new Error(`navegación fallida: ${navigation.errorText}`);
    }
    await loaded;

    // Espera a que fuentes y layout se estabilicen, igual que hace browser-check.
    await run(`(function () {
      document.documentElement.setAttribute('data-motion', 'still');
      var fontsReady = document.fonts && document.fonts.ready
        ? document.fonts.ready.catch(function () {})
        : Promise.resolve();
      return fontsReady.then(function () {
        if (window.Archify && Archify.layoutStability && typeof Archify.layoutStability.whenStable === 'function') {
          return Archify.layoutStability.whenStable();
        }
      }).then(function () {
        return new Promise(function (resolve) {
          requestAnimationFrame(function () { requestAnimationFrame(resolve); });
        });
      });
    })()`);

    // Intercepta el blob que produce la exportación en lugar de descargarlo.
    await run(`(function () {
      window.__archifySvg = null;
      var originalCreate = URL.createObjectURL;
      URL.createObjectURL = function (blob) {
        if (blob && blob.type && blob.type.indexOf('image/svg+xml') === 0) {
          window.__archifySvg = blob.text();
        }
        return originalCreate.call(URL, blob);
      };
      return true;
    })()`, false);

    const clicked = await run(`(function () {
      var button = document.querySelector('button[data-format="svg-light"]');
      if (!button) return false;
      button.click();
      return true;
    })()`, false);
    if (!clicked) throw new Error("no se encontró el botón de exportación SVG");

    // resvg (el motor SVG de Typst) no implementa las propiedades personalizadas
    // de CSS, así que `var(--x)` se resolvería a negro. Sustituimos cada una por
    // el valor que el propio navegador calcula con el tema claro aplicado.
    const svg = await run(`(function () {
      return Promise.resolve(window.__archifySvg).then(function (text) {
        if (!text) return null;
        var roots = [document.documentElement, document.querySelector('svg[data-theme]')].filter(Boolean);
        var names = text.match(/--[a-zA-Z0-9-]+/g) || [];
        var seen = {};
        names.forEach(function (name) {
          if (seen[name] !== undefined) return;
          var value = '';
          for (var i = 0; i < roots.length && !value; i += 1) {
            value = getComputedStyle(roots[i]).getPropertyValue(name).trim();
          }
          seen[name] = value;
        });
        var pattern = /var\\(\\s*(--[a-zA-Z0-9-]+)\\s*(?:,\\s*([^()]*?)\\s*)?\\)/g;
        var unresolved = [];
        for (var pass = 0; pass < 8; pass += 1) {
          var changed = false;
          text = text.replace(pattern, function (whole, name, fallback) {
            var value = seen[name];
            if (value) { changed = true; return value; }
            if (fallback) { changed = true; return fallback; }
            unresolved.push(name);
            return whole;
          });
          if (!changed) break;
        }
        return { svg: text, unresolved: Array.from(new Set(unresolved)) };
      });
    })()`);
    if (!svg || !svg.svg) throw new Error("la exportación SVG no produjo contenido");
    if (svg.unresolved.length) {
      console.warn(`  aviso, variables sin resolver: ${svg.unresolved.join(", ")}`);
    }

    fs.mkdirSync(path.dirname(svgPath), { recursive: true });
    fs.writeFileSync(svgPath, svg.svg, "utf8");
    console.log(
      `${path.basename(svgPath)}: ${(svg.svg.length / 1024).toFixed(1)} KiB`,
    );
  }
} finally {
  browser.close();
}
