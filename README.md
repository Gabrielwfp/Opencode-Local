# Guía completa: llama.cpp + llama-swap + 3 modelos locales en Mac Studio M4 64GB

> **Hardware objetivo:** Mac Studio M4, 64 GB Unified Memory
> **Modelos:** Qwen3.6-35B-A3B · Qwen3.5-9B · Gemma-4-26B-A4B
> **Objetivo final:** Un único endpoint OpenAI-compatible en el puerto 8080,
> gestionado por llama-swap, que carga/descarga modelos bajo demanda.
> Listo para usarse con oh-my-openagent u otras herramientas.

---

## Dos modos de operación — elige el tuyo

| | Modo A: Servidores en paralelo | Modo B: llama-swap ⭐ recomendado |
|---|---|---|
| **Cómo funciona** | 3 procesos corriendo siempre | 1 proxy que carga el modelo cuando lo necesita |
| **RAM usada** | ~43 GB constantes | Solo el modelo activo (~5–20 GB) |
| **Latencia de cambio** | 0 ms (ya cargado) | 15–30 s (carga bajo demanda) |
| **Un solo endpoint** | No (3 puertos distintos) | ✅ Sí (puerto 8080 único) |
| **Ideal para** | Benchmarks, pipelines paralelos | Uso interactivo normal y desarrollo |

> Para el uso típico con oh-my-openagent (los agentes se turnan, no actúan en paralelo),
> **llama-swap es la opción superior**: ahorra ~25 GB de RAM y simplifica toda la configuración.
> La guía cubre **ambos modos**: instala todo el stack base y al final eliges.

---

## Índice

1. [Requisitos previos del sistema](#1-requisitos-previos-del-sistema)
2. [Instalar Homebrew](#2-instalar-homebrew)
3. [Instalar herramientas de compilación](#3-instalar-herramientas-de-compilación)
4. [Clonar y compilar llama.cpp con Metal](#4-clonar-y-compilar-llamacpp-con-metal)
5. [Instalar Python y huggingface-cli](#5-instalar-python-y-huggingface-cli)
6. [Crear estructura de carpetas](#6-crear-estructura-de-carpetas)
7. [Descargar los modelos](#7-descargar-los-modelos)
8. [Verificar los modelos con un test rápido](#8-verificar-los-modelos-con-un-test-rápido)
9. [Modo A — Lanzar tres servidores en paralelo](#9-modo-a--lanzar-tres-servidores-en-paralelo)
10. [Modo B — Instalar y configurar llama-swap ⭐](#10-modo-b--instalar-y-configurar-llama-swap-)
11. [Speculative Decoding — Qwen3.6 más rápido ⚡](#11-speculative-decoding--qwen36-más-rápido-)
12. [Automatizar con launchd (inicio automático)](#12-automatizar-con-launchd-inicio-automático)
13. [Configurar oh-my-openagent](#13-configurar-oh-my-openagent)
14. [Verificación final](#14-verificación-final)
15. [Referencia rápida de comandos](#15-referencia-rápida-de-comandos)
16. [Solución de problemas frecuentes](#16-solución-de-problemas-frecuentes)

---

## 1. Requisitos previos del sistema

### 1.1 Verifica tu chip y memoria

Abre Terminal (`Cmd + Espacio` → escribe `Terminal`) y ejecuta:

```bash
# Confirma que es Apple Silicon (arm64)
uname -m
# Resultado esperado: arm64

# Verifica memoria unificada
system_profiler SPHardwareDataType | grep "Memory:"
# Resultado esperado: Memory: 64 GB

# Verifica soporte Metal
system_profiler SPDisplaysDataType | grep Metal
# Resultado esperado: Metal: Supported, feature set macOS GPUFamily2 v1 (o superior)
```

> **¿Por qué Apple Silicon es eficiente para LLMs?**
> La arquitectura de **memoria unificada** de los chips M-series elimina la copia de datos
> entre CPU y GPU — en hardware convencional (NVIDIA/AMD), cada inferencia implica
> transferir pesos del sistema RAM a VRAM y viceversa. En M4, CPU, GPU y Neural Engine
> comparten el mismo pool de 64 GB físicos. Ese zero-copy overhead es la razón principal
> por la que llama.cpp alcanza 40–60 t/s en modelos de 22 GB en este hardware.

### 1.2 Verifica versión de macOS

```bash
sw_vers
# Necesitas macOS 13 Ventura o superior para Metal estable con llama.cpp
# Recomendado: macOS 14 Sonoma o macOS 15 Sequoia
```

Si tu macOS está desactualizado, ve a **Preferencias del Sistema → General → Actualización de Software**.

### 1.3 Espacio en disco necesario

| Modelo | Tamaño en disco |
|--------|----------------|
| Qwen3.6-35B-A3B UD-Q4_K_M | ~22 GB |
| Qwen3.5-2B Q4_K_M (draft) | ~1.7 GB |
| Qwen3.5-9B Q4_K_M | ~5 GB |
| Gemma-4-26B-A4B UD-Q4_K_XL | ~16 GB |
| Gemma-4-26B-A4B mmproj-BF16 (visión) | ~2 GB |
| llama.cpp (compilado) | ~500 MB |
| **Total** | **~47 GB** |

```bash
df -h ~
# Necesitas al menos 60 GB libres
```

---

## 2. Instalar Homebrew

Homebrew es el gestor de paquetes para macOS.

### 2.1 Instalar Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

El instalador te pedirá tu contraseña de administrador. Tarda 2–5 minutos.

### 2.2 Configurar Homebrew en el PATH

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### 2.3 Verificar instalación

```bash
brew --version
# Resultado esperado: Homebrew 4.x.x

which brew
# Resultado esperado: /opt/homebrew/bin/brew
# ⚠️  Si ves /usr/local/bin/brew tienes la versión Intel — reinstala con el comando de arriba
```

---

## 3. Instalar herramientas de compilación

### 3.1 Xcode Command Line Tools

```bash
xcode-select --install
```

Aparecerá una ventana gráfica — haz clic en **Instalar**. Tarda ~10–15 minutos aunque diga "20 horas".

```bash
xcode-select -p
# Resultado esperado: /Library/Developer/CommandLineTools
```

### 3.2 CMake y Git

```bash
brew install cmake git
```

```bash
cmake --version   # cmake version 3.x.x (necesitas 3.14 o superior)
git --version     # git version 2.x.x
```

---

## 4. Clonar y compilar llama.cpp con Metal

### 4.1 Clonar el repositorio

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

### 4.2 Compilar con soporte Metal (GPU Apple Silicon)

```bash
cmake -B build \
  -DGGML_METAL=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

> `-j$(sysctl -n hw.ncpu)` usa todos los núcleos disponibles. El proceso tarda 3–8 minutos.

### 4.3 Verificar compilación

```bash
ls ~/llama.cpp/build/bin/
# Debes ver: llama-cli  llama-server  llama-bench  llama-mtmd-cli  (entre otros)

# Verificar que el build soporta los flags de reasoning (necesario para los tres modelos)
~/llama.cpp/build/bin/llama-server --help 2>&1 | grep -i "reasoning\|think"
# Debe mostrar:
# -rea, --reasoning [on|off|auto]
# --reasoning-format FORMAT
# --reasoning-budget N
```

### 4.4 Agregar los binarios al PATH

```bash
echo 'export PATH="$HOME/llama.cpp/build/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

which llama-server
# Resultado esperado: /Users/tu-usuario/llama.cpp/build/bin/llama-server
```

---

## 5. Instalar hf CLI (Hugging Face)

El CLI `hf` es la herramienta oficial para descargar modelos desde Hugging Face.
Tienes dos opciones — elige la más simple para ti:

### Opción A — Instalador standalone (sin Python, recomendado)

Un solo comando, no requiere tener Python instalado:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Cierra y vuelve a abrir el terminal, luego verifica:

```bash
hf --version
# hf x.x.x
```

### Opción B — Homebrew (ya tienes Homebrew instalado)

```bash
brew install huggingface-cli

hf --version
# hf x.x.x
```

### Opción C — pip (solo si necesitas Python para otros fines)

```bash
brew install python@3.11
pip3.11 install --upgrade huggingface_hub

hf --version
# huggingface_hub/0.x.x
```

> **Nota:** Las opciones A y B instalan el binario `hf` directamente, sin necesidad
> de Python. La opción C instala el paquete Python `huggingface_hub` que también
> incluye el comando `hf` (antes llamado `huggingface-cli`, ambos funcionan).

---

## 6. Crear estructura de carpetas

```bash
mkdir -p ~/llm-models/qwen3.6-35b
mkdir -p ~/llm-models/qwen3.5-2b
mkdir -p ~/llm-models/qwen3.5-9b
mkdir -p ~/llm-models/gemma4-26b
mkdir -p ~/llm-servers

ls ~/llm-models/
# qwen3.5-2b  qwen3.6-35b  gemma4-26b  qwen3.5-9b
```

---

## 7. Descargar los modelos

> ⏱️ **Tiempo estimado total:** 2–4 horas dependiendo de tu conexión a internet.
> Los tres modelos suman ~47 GB (incluyendo el draft model Qwen3.5-2B).

Tienes dos métodos equivalentes:

**Método A — `hf download`** — recomendado, reanuda descargas interrumpidas, soporte `--dry-run`
**Método B — `llama-server -hf`** — descarga y lanza directamente, sin CLI

> **Tip — verificar antes de descargar:** usa `--dry-run` para ver qué archivos
> se descargarán y su tamaño, sin iniciar la descarga real:
> ```bash
> hf download unsloth/Qwen3.6-35B-A3B-GGUF --dry-run
> ```

---

### 7.1 Modelo 1 — Qwen3.6-35B-A3B (Razonamiento)

**~22 GB modelo principal + ~1.7 GB draft model**

Este modelo usa **thinking mode** controlado por el chat template con `--jinja`.
Qwen3.6-35B es arquitectura MoE (35B parámetros totales, 3.6B activos por token) —
muy eficiente en inferencia a pesar del tamaño total del modelo.

El **draft model** (2B) se descarga junto al principal — necesario para activar
Speculative Decoding en la sección 11 y lograr hasta 2x más velocidad de inferencia.
Ambos comparten el mismo tokenizer Qwen, requisito indispensable para que funcione.

```bash
# Verificar qué se descargará antes de ejecutar (~22 GB)
hf download unsloth/Qwen3.6-35B-A3B-GGUF --dry-run

# Modelo principal
hf download \
  unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --local-dir ~/llm-models/qwen3.6-35b

ls -lh ~/llm-models/qwen3.6-35b/
# Qwen3.6-35B-A3B-UD-Q4_K_M.gguf  (~22 GB)

# Draft model para Speculative Decoding (~1.7 GB extra)
hf download \
  unsloth/Qwen3.5-2B-GGUF \
  Qwen3.5-2B-Q4_K_M.gguf \
  --local-dir ~/llm-models/qwen3.5-2b

ls -lh ~/llm-models/qwen3.5-2b/
# Qwen3.5-2B-Q4_K_M.gguf  (~1.7 GB)
```

```bash
# Método B — descarga directa con llama-server (sin CLI)
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M \
  -ngl 99 -c 65536 --port 8080
# El modelo queda en ~/.cache/llama.cpp/
# Para llama-swap, copia el archivo a ~/llm-models/qwen3.6-35b/
```

> **¿Por qué este modelo?** Arquitectura MoE con 35B parámetros totales (3.6B activos),
> muy eficiente en inferencia. Thinking mode controlado vía chat template con `--jinja`.
> Ideal para Sisyphus, Oracle y Prometheus.
> Parámetros óptimos: `temp=0.6, top_p=0.95, top_k=20, min_p=0.0`.

---

### 7.2 Modelo 2 — Qwen3.5-9B (Velocidad)

**~5 GB**

Qwen3.5-9B tiene **thinking desactivado por defecto** — los modelos Small (0.8B–9B)
no activan thinking por defecto según la documentación oficial de Unsloth.
Se configura con `-rea off` como medida adicional de seguridad.

```bash
# Verificar antes de descargar
hf download unsloth/Qwen3.5-9B-GGUF --dry-run

# Descargar
hf download \
  unsloth/Qwen3.5-9B-GGUF \
  Qwen3.5-9B-Q4_K_M.gguf \
  --local-dir ~/llm-models/qwen3.5-9b

ls -lh ~/llm-models/qwen3.5-9b/
# Qwen3.5-9B-Q4_K_M.gguf  (~5 GB)
```

> **¿Por qué este modelo?** Compacto y muy rápido (~90 t/s en M4). Ideal para los
> agentes utilitarios Explore y Librarian que hacen búsquedas rápidas en código.
> Parámetros óptimos modo instruct (Unsloth): `temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.

---

### 7.3 Modelo 3 — Gemma-4-26B-A4B (Creatividad / Multimodal)

**~16 GB modelo + ~2 GB mmproj (visión)**

Gemma 4 también tiene **thinking mode** — se desactiva con
`--chat-template-kwargs '{"enable_thinking":false}'` en `llama-server`.
Para visión multimodal se necesita el archivo `mmproj-BF16.gguf` adicional.
Se descarga el formato `UD-Q4_K_XL` (Dynamic Unsloth) en lugar de Q4_K_M estándar —
sube a 8-bit las capas más críticas, manteniendo mayor precisión con el mismo tamaño.

```bash
# Verificar antes de descargar (incluye mmproj ~2 GB)
hf download unsloth/gemma-4-26B-A4B-it-GGUF --dry-run

# Descargar modelo UD-Q4_K_XL + mmproj para visión
hf download \
  unsloth/gemma-4-26B-A4B-it-GGUF \
  --local-dir ~/llm-models/gemma4-26b \
  --include "*UD-Q4_K_XL*" \
  --include "*mmproj-BF16*"

ls -lh ~/llm-models/gemma4-26b/
# gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf  (~16 GB)
# mmproj-BF16.gguf                      (~2 GB)
```

> **¿Por qué este modelo?** Diseñado por Google para workflows agénticos, multimodal
> (imagen + texto), coding y long-context con 256K tokens de contexto.
> Parámetros oficiales Google/Unsloth: `temp=1.0, top_p=0.95, top_k=64`.

---

### 7.4 Verificar los tres modelos + draft

```bash
echo "=== Modelos descargados ==="
ls -lh ~/llm-models/qwen3.6-35b/
ls -lh ~/llm-models/qwen3.5-2b/
ls -lh ~/llm-models/qwen3.5-9b/
ls -lh ~/llm-models/gemma4-26b/

echo "=== Espacio total ==="
du -sh ~/llm-models/
```

---

## 8. Verificar los modelos con un test rápido

Antes de configurar los servidores, valida que cada modelo cargue y que Metal se active.
Usa `-n 2048` — suficientes tokens para que el modelo complete la respuesta incluso con thinking.

### 8.1 Test — Qwen3.6-35B-A3B

```bash
llama-cli \
  -m ~/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 \
  -n 2048 \
  -p "Explica en una oración qué es la inteligencia artificial." \
  --no-display-prompt
```

En la salida inicial debes ver:
```
ggml_metal_init: GPU name: Apple M4
llm_load_tensors: offloaded 64/64 layers to GPU
```

Verás el bloque `[Start thinking]` seguido de la respuesta real — comportamiento
correcto para este modelo. En producción con llama-swap y `--jinja`, el thinking
queda controlado por el chat template embebido en el GGUF.

### 8.2 Test — Qwen3.5-9B

```bash
llama-cli \
  -m ~/llm-models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -n 200 \
  -p "Hola, ¿quién eres?" \
  --no-display-prompt
```

> Si ves `[Start thinking]` con este modelo en llama-cli, no es un error —
> llama-cli no siempre aplica el chat template correctamente. En llama-server
> con `--jinja -rea off` el thinking queda completamente suprimido.

### 8.3 Test — Gemma-4-26B-A4B (solo texto)

```bash
llama-cli \
  -m ~/llm-models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  -ngl 99 \
  -n 200 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  -p "Describe the sky in one sentence." \
  --no-display-prompt
```

> Si ves `[Start thinking]`, es normal en llama-cli. En llama-server con
> `--jinja --chat-template-kwargs '{"enable_thinking":false}'` se desactiva.

---

## 9. Modo A — Lanzar tres servidores en paralelo

> Usa este modo solo si necesitas que los tres modelos respondan **simultáneamente**.
> Para uso normal con oh-my-openagent, ve directamente al **Modo B — sección 10**.

### 9.1 Servidor 1 — Qwen3.6-35B-A3B (Puerto 8080)

```bash
llama-server \
  -m ~/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 \
  -c 65536 \
  --host 127.0.0.1 \
  --port 8080 \
  --flash-attn on \
  --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --no-webui
```

Espera: `llama server listening at http://127.0.0.1:8080`

> **Flags clave:**
> - `-fa` — Flash Attention (alias de `--flash-attn on`)
> - `--jinja` — activa el chat template embebido en el GGUF (controla thinking mode)

### 9.2 Servidor 2 — Qwen3.5-9B (Puerto 8081)

```bash
llama-server \
  -m ~/llm-models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 65536 \
  --host 127.0.0.1 \
  --port 8081 \
  --flash-attn on \
  --jinja \
  -rea off \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 \
  --no-webui
```

> `-rea off` — desactiva thinking completamente (confirmado funcional en build 8738+)

### 9.3 Servidor 3 — Gemma-4-26B-A4B (Puerto 8082)

```bash
llama-server \
  -m ~/llm-models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --mmproj ~/llm-models/gemma4-26b/mmproj-BF16.gguf \
  -ngl 99 \
  -c 65536 \
  --host 127.0.0.1 \
  --port 8082 \
  --flash-attn on \
  -ctk q8_0 -ctv q8_0 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --no-webui
```

> **Flags clave de Gemma 4:**
> - `--mmproj` — proyector multimodal para visión
> - `-ctk q8_0 -ctv q8_0` — previene freeze con Flash Attention en Apple Silicon
> - `--chat-template-kwargs '{"enable_thinking":false}'` — método canónico para desactivar thinking

### 9.4 Verificar los tres servidores

```bash
for PORT in 8080 8081 8082; do
  STATUS=$(curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if echo "$STATUS" | grep -qi "ok"; then
    echo "✅ Puerto $PORT: OK"
  else
    echo "❌ Puerto $PORT: no responde"
  fi
done
```

---

## 10. Modo B — Instalar y configurar llama-swap ⭐

llama-swap es un proxy ligero escrito en Go — un solo binario, un archivo YAML — que
se sienta frente a `llama-server` y carga el modelo correcto cuando llega una petición.
Si el modelo pedido es diferente al que está cargado, lo intercambia automáticamente.

### ¿Por qué llama-swap mejora esta guía?

- **Un solo endpoint** — `http://127.0.0.1:8080` para los tres modelos
- **Carga bajo demanda** — solo el modelo activo ocupa RAM (~5–20 GB en vez de ~43 GB)
- **TTL automático** — descarga el modelo si no recibe peticiones en X segundos
- **Sin administrar puertos** — oh-my-openagent habla con un solo provider
- **Hot-reload de config** — cambias el YAML y llama-swap recarga sin reiniciar
- **UI web integrada** — monitoreo, logs y playground en `localhost:8080`

### 10.1 Instalar llama-swap

```bash
brew tap mostlygeek/llama-swap
brew install llama-swap

llama-swap --version
# version: vXXX (hash), built at YYYY-MM-DD
```

### 10.2 Crear el archivo de configuración

```bash
cat > ~/llm-servers/config.yaml << 'EOF'
# yaml-language-server: $schema=https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/config-schema.json
# llama-swap config — Mac Studio M4 64GB

healthCheckTimeout: 120
globalTTL: 1200
logLevel: info

models:

  # ── Razonamiento (≡ GPT) ─────────────────────────────────
  # Qwen3.6-35B: MoE 35B parámetros, 3.6B activos.
  # Thinking mode controlado por chat template con --jinja.
  # --model-draft activa Speculative Decoding con el modelo 2B (mismo tokenizer Qwen).
  # Beneficio: hasta 2x más tokens/s en los bloques de thinking extensos.
  # --reasoning-budget 4096: limita el bloque <think> a 4K tokens; sin límite (-1 default)
  #   los bloques de pensamiento llenan el contexto y provocan bucles en sesiones largas.
  # -c 131072: contexto extendido a 128K (~46GB total con KV cache); permite sesiones largas
  #   sin desbordamiento. opencode.json lo declara en 98304 dejando 32K de margen.
  "qwen3.6-35b":
    name: "Qwen3.6 35B"
    description: "Razonamiento profundo, cadena de pensamiento"
    cmd: |
      /Users/TUNOMBRE/llama.cpp/build/bin/llama-server \
        --port ${PORT} \
        -m /Users/TUNOMBRE/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        --model-draft /Users/TUNOMBRE/llm-models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf \
        --draft 32 \
        --draft-min 5 \
        --draft-p-min 0.9 \
        -ngl 99 \
        -ngld 99 \
        -c 131072 \
        --flash-attn on \
        --jinja \
        --no-webui \
        --temp 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --min-p 0.0 \
        --reasoning-budget 4096
    ttl: 1800
    filters:
      setParams:
        max_tokens: 8192
    aliases:
      - "gpt-4o"
      - "claude-opus-4-6"

  # ── Velocidad (≡ Minimax) ─────────────────────────────────
  # Qwen3.5-9B: thinking desactivado por defecto en modelos Small (0.8B-9B).
  # -rea off como garantía adicional.
  # Parámetros oficiales Unsloth para modo instruct (non-thinking).
  "qwen3.5-9b":
    name: "Qwen3.5 9B"
    description: "Tareas rápidas, búsquedas en código"
    cmd: |
      /Users/TUNOMBRE/llama.cpp/build/bin/llama-server \
        --port ${PORT} \
        -m /Users/TUNOMBRE/llm-models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
        -ngl 99 \
        -c 65536 \
        --flash-attn on \
        --jinja \
        -rea off \
        --no-webui \
        --temp 0.7 \
        --top-p 0.8 \
        --top-k 20 \
        --min-p 0.0 \
        --presence-penalty 1.5
    ttl: 900
    aliases:
      - "gpt-4o-mini"
      - "gpt-3.5-turbo"

  # ── Creatividad / Multimodal (≡ Gemini) ───────────────────
  # Gemma 4 tiene thinking mode — se desactiva con chat-template-kwargs.
  # --mmproj habilita visión multimodal (imagen + texto).
  # -ctk q8_0 -ctv q8_0 previene freeze con Flash Attention en Apple Silicon.
  # Parámetros oficiales Google/Unsloth: temp=1.0, top_p=0.95, top_k=64.
  "gemma4-26b":
    name: "Gemma 4 26B A4B IT"
    description: "Creatividad, visión, frontend, multimodal"
    cmd: |
      /Users/TUNOMBRE/llama.cpp/build/bin/llama-server \
        --port ${PORT} \
        -m /Users/TUNOMBRE/llm-models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
        --mmproj /Users/TUNOMBRE/llm-models/gemma4-26b/mmproj-BF16.gguf \
        -ngl 99 \
        -c 65536 \
        --flash-attn on \
        -ctk q8_0 \
        -ctv q8_0 \
        --jinja \
        --chat-template-kwargs '{"enable_thinking":false}' \
        --no-webui \
        --temp 1.0 \
        --top-p 0.95 \
        --top-k 64
    ttl: 1800
    filters:
      setParams:
        max_tokens: 2048
    aliases:
      - "gemini-flash"
EOF
```

Ahora sustituye `TUNOMBRE` por tu usuario real:

```bash
sed -i '' "s/TUNOMBRE/$(whoami)/g" ~/llm-servers/config.yaml

# Verificar que quedó bien
grep "llm-models" ~/llm-servers/config.yaml
# Debe mostrar rutas como: /Users/user/llm-models/...
```

### 10.3 Entender el config.yaml

| Campo | Qué hace |
|---|---|
| `${PORT}` | llama-swap asigna un puerto libre automáticamente a cada proceso |
| `globalTTL: 1200` | TTL por defecto: descarga modelos inactivos a los 20 min |
| `ttl: 1800` | Este modelo se descarga 30 min después de la última petición |
| `healthCheckTimeout: 120` | Espera hasta 120 s para que el modelo arranque |
| `--flash-attn on` | Flash Attention — reduce uso de memoria del KV cache (forma explícita requerida en Apple Silicon) |
| `-ctk q8_0 -ctv q8_0` | Cache type para prevenir freeze en Apple Silicon con Gemma 4 |
| `--jinja` | Activa el chat template embebido en el GGUF (controla thinking mode en Qwen3.6) |
| `-rea off` | Desactiva thinking completamente (Qwen3.5-9B) |
| `--chat-template-kwargs '{"enable_thinking":false}'` | Desactiva thinking en Gemma 4 (solo funciona en llama-server con --jinja) |
| `--mmproj` | Proyector multimodal para visión en Gemma 4 |
| `--model-draft` | Ruta al modelo draft para Speculative Decoding (debe tener el mismo tokenizer) |
| `--draft 16` | Tokens que propone el draft por paso — 16 es el valor recomendado |
| `--draft-min 5` | Mínimo de tokens draft antes de verificar con el modelo principal |
| `--draft-p-min 0.9` | Descarta drafts de baja confianza temprano — reduce verificaciones inútiles |
| `-ngld 99` | GPU layers para el draft model (separado de `-ngl` del modelo principal) |
| `filters.setParams.max_tokens` | Sobreescribe max_tokens — garantiza tokens suficientes para que thinking termine |
| `aliases` | Nombres alternativos: peticiones con `"model":"gpt-4o"` → `qwen3.6-35b` |

### 10.4 Test manual antes de automatizar

```bash
# Lanzar llama-swap en primer plano (Ctrl+C para detener)
llama-swap \
  --config ~/llm-servers/config.yaml \
  --listen 127.0.0.1:8080

# En otra pestaña — verificar que el proxy responde
curl -s http://127.0.0.1:8080/health
# Respuesta: OK

# Ver modelos registrados
curl -s http://127.0.0.1:8080/v1/models | python3.11 -m json.tool
# Lista vacía al inicio — los modelos se cargan bajo demanda

# Primera petición — llama-swap arranca el modelo automáticamente
# Puede tardar 15–30 s la primera vez
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Responde solo: hola"}],
    "max_tokens": 50
  }' | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
print('Respuesta:', d['choices'][0]['message']['content'])
"
# En la consola de llama-swap verás:
# loading model "qwen3.5-9b"
# model "qwen3.5-9b" ready on port XXXXX
```

### 10.5 Verificar los tres modelos y el thinking

```bash
# Qwen3.6-35B — verificar respuesta y thinking mode
echo "=== Qwen3.6-35B ==="
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Explica en una oración qué es la IA."}],
    "max_tokens": 500
  }' | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
print('RESPUESTA:', msg['content'])
print('THINKING (primeros 80 chars):', msg.get('reasoning_content','(no sep — controlado por jinja)')[:80])
"

# Qwen3.5-9B — sin thinking en content ni reasoning_content
echo "=== Qwen3.5-9B ==="
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Di hola en una palabra."}],
    "max_tokens": 50
  }' | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
print('RESPUESTA:', msg['content'])
print('THINKING:', msg.get('reasoning_content','(vacío — correcto)'))
"

# Gemma 4 — sin thinking
echo "=== Gemma4-26B ==="
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-26b",
    "messages": [{"role": "user", "content": "Di hola en una palabra."}],
    "max_tokens": 50
  }' | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
print('RESPUESTA:', msg['content'])
print('THINKING:', msg.get('reasoning_content','(vacío — correcto)'))
"
```

### 10.6 Interfaz web integrada de llama-swap

```bash
open http://127.0.0.1:8080
```

Verás qué modelo está activo, uso de memoria, historial de peticiones y un playground
para probar los modelos manualmente.

### 10.7 Endpoints de administración

```bash
# Ver modelo actualmente cargado en RAM
curl -s http://127.0.0.1:8080/running | python3.11 -m json.tool
# {
#   "running": [{
#     "model": "qwen3.5-9b",
#     "state": "ready",
#     "proxy": "http://localhost:5800",
#     "ttl": 900
#   }]
# }

# Descargar todos los modelos manualmente (liberar RAM)
curl -s http://127.0.0.1:8080/unload
# Respuesta: OK

# Acceso directo al upstream de un modelo específico
curl -s http://127.0.0.1:8080/upstream/qwen3.5-9b/health
curl -s http://127.0.0.1:8080/upstream/qwen3.5-9b/slots

# Streaming de logs en tiempo real
curl -sN http://127.0.0.1:8080/logs/stream                    # todos
curl -sN http://127.0.0.1:8080/logs/stream/proxy              # solo proxy
curl -sN http://127.0.0.1:8080/logs/stream/qwen3.6-35b    # solo un modelo
curl -sN 'http://127.0.0.1:8080/logs/stream?no-history'       # solo logs nuevos

# Monitoreo de tokens por segundo
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","stream":true,"timings_per_token":true,
       "messages":[{"role":"user","content":"Escribe 3 oraciones cortas."}]}' \
  | grep -o '"predicted_per_second":[0-9.]*'
```

---

## 11. Speculative Decoding — Qwen3.6 más rápido ⚡

Speculative Decoding es una técnica que acelera la inferencia sin cambiar la calidad
de la salida. Usa un modelo pequeño ("draft") para proponer tokens candidatos, y el
modelo grande los verifica en paralelo. El resultado es idéntico al obtenido sin la
técnica, pero significativamente más rápido.

**¿Por qué es especialmente útil para Qwen3.6-35B en tu setup?**

Qwen3.6-35B genera bloques de thinking extensos antes de responder — cientos o miles
de tokens de razonamiento interno. Speculative Decoding tiene más margen de ganancia
cuanto más larga es la salida, lo que lo convierte en la optimización más impactante
para este modelo. En Apple Silicon, la latencia de carga del draft model es mínima
gracias a la memoria unificada.

### Requisito fundamental

El draft model **debe tener exactamente el mismo tokenizer** que el modelo principal.
Por eso `Qwen3.5-2B` es el candidato correcto para `Qwen3.6-35B-A3B` — ambos
comparten el tokenizer Qwen.

### 11.1 Verificar que el draft model está descargado

```bash
ls -lh ~/llm-models/qwen3.5-2b/
# Qwen3.5-2B-Q4_K_M.gguf  (~1.7 GB)
# Si no está, descárgalo: hf download unsloth/Qwen3.5-2B-GGUF Qwen3.5-2B-Q4_K_M.gguf --local-dir ~/llm-models/qwen3.5-2b
```

### 11.2 Test de Speculative Decoding en primer plano

Antes de confiar en llama-swap, prueba que el draft model funciona correctamente:

```bash
llama-server \
  -m ~/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --model-draft ~/llm-models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf \
  --draft 16 \
  --draft-min 5 \
  --draft-p-min 0.9 \
  -ngl 99 \
  -ngld 99 \
  -c 65536 \
  --flash-attn on \
  --jinja \
  --port 9999 \
  --no-webui
```

En otra pestaña, mide la velocidad con y sin draft para comparar:

```bash
# Test CON Speculative Decoding (puerto 9999)
time curl -s http://127.0.0.1:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "Explica paso a paso cómo funciona la fotosíntesis."}],
    "max_tokens": 1000,
    "stream": false
  }' | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
usage=d.get('usage',{})
print('Tokens generados:', usage.get('completion_tokens','?'))
print('Respuesta (primeros 100 chars):', d['choices'][0]['message']['content'][:100])
"
```

En los logs del servidor verás líneas como:
```
draft acceptance rate: 0.72   ← 72% de tokens del draft aceptados por el modelo grande
speculative tokens per step: 11.5
```

Una acceptance rate >0.6 indica que la aceleración es efectiva.

### 11.3 Parámetros del config.yaml (ya integrados en sección 10.2)

El `config.yaml` de la sección 10.2 ya incluye `--model-draft` y `--draft 16` en el
modelo `qwen3.6-35b`. Los parámetros clave son:

| Flag | Valor recomendado | Qué hace |
|---|---|---|
| `--model-draft` | ruta al 2B Q4_K_M | Modelo draft — debe tener el mismo tokenizer que el principal |
| `--draft` | `16` | Tokens que propone el draft por paso |
| `--draft-min` | `5` | Mínimo de tokens draft antes de ceder al modelo principal |
| `--draft-p-min` | `0.9` | Descarta drafts de baja confianza temprano; bájalo si acceptance rate >0.85 |
| `-ngld` | `99` | GPU layers para el draft model — separado de `-ngl` del modelo principal |

> `--draft 16` es el valor de referencia. Prueba con `8` (más conservador) o `32` (más agresivo)
> según el acceptance rate que obtengas. Si el rate cae <0.5, bajar a `8–12`.
> `-ngld 99` garantiza que el draft model corre en Metal, no en CPU.

### 11.4 Qué esperar en tu hardware

| Escenario | Tokens/s esperados en M4 64GB |
|---|---|
| Sin speculative decoding | ~25–30 t/s |
| Con speculative decoding (draft 16) | ~40–55 t/s |
| Acceptance rate típica en reasoning | 0.60–0.75 |

El mayor beneficio se observa en los bloques de `<think>` — el razonamiento interno
tiende a ser más predecible que el texto creativo, lo que eleva la acceptance rate.

### 11.5 Cuándo desactivarlo

Si encuentras que la acceptance rate es baja (<0.4) o el comportamiento del modelo
cambia, elimina `--model-draft` y `--draft 16` del config.yaml. llama-swap recargará
la configuración automáticamente con `--watch-config`.

```bash
# Verificar acceptance rate en tiempo real
curl -sN http://127.0.0.1:8080/logs/stream/qwen3.6-35b | grep -i "draft\|speculative"
```

---

## 12. Automatizar con launchd (inicio automático)

launchd es el sistema de servicios de macOS, equivalente a systemd en Linux.

### 12A. launchd para llama-swap (Modo B — recomendado)

```bash
MY_HOME=$HOME

cat > ~/Library/LaunchAgents/com.llmserver.llamaswap.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.llmserver.llamaswap</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/llama-swap</string>
    <string>--config</string>
    <string>${MY_HOME}/llm-servers/config.yaml</string>
    <string>--listen</string>
    <string>127.0.0.1:8080</string>
    <string>--watch-config</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${MY_HOME}/llm-servers/llamaswap.log</string>
  <key>StandardErrorPath</key>
  <string>${MY_HOME}/llm-servers/llamaswap-err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>ExitTimeout</key>
  <integer>60</integer>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>ProcessType</key>
  <string>Background</string>
</dict>
</plist>
EOF

# Activar el servicio (macOS Ventura+ — usar bootstrap, no load)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist

# Verificar
launchctl list | grep llamaswap
# Debes ver un PID en la primera columna

sleep 3
curl -s http://127.0.0.1:8080/health
# Respuesta: OK
```

> `--watch-config` hace que llama-swap recargue `config.yaml` automáticamente cuando
> lo edites — útil para agregar modelos sin reiniciar el servicio.
>
> **Keys del plist:**
> - `ExitTimeout: 60` — espera 60 s antes de SIGKILL al detener; el default de 20 s
>   puede cortar un unload de modelo a mitad (llama-swap necesita liberar ~22 GB de RAM).
> - `ThrottleInterval: 30` — mínimo 30 s entre reinicios automáticos; evita bucles
>   rápidos si llama-swap falla al arrancar (ej. GGUF no encontrado).
> - `ProcessType: Background` — hint al scheduler de macOS para tratar el proceso
>   como servicio de fondo, sin impactar la prioridad de apps interactivas.

**Comandos de control:**

```bash
# API moderna — obligatoria en macOS Ventura+ (load/unload da errno 5)
launchctl bootout  gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist
launchctl kickstart -k gui/$(id -u)/com.llmserver.llamaswap   # restart rápido

# Debug detallado del servicio (estado, PID, exit code, variables de entorno)
launchctl print gui/$(id -u)/com.llmserver.llamaswap

tail -f ~/llm-servers/llamaswap.log       # logs
tail -f ~/llm-servers/llamaswap-err.log   # errores
```

---

### 12B. launchd para servidores en paralelo (Modo A)

```bash
# Script Qwen3.6-35B
cat > ~/llm-servers/start-qwen3.6.sh << SCRIPT
#!/bin/bash
exec /Users/$(whoami)/llama.cpp/build/bin/llama-server \
  -m /Users/$(whoami)/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --model-draft /Users/$(whoami)/llm-models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf \
  --draft 16 --draft-min 5 --draft-p-min 0.9 \
  -ngl 99 -ngld 99 -c 65536 --host 127.0.0.1 --port 8080 \
  --flash-attn on --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --no-webui
SCRIPT

# Script Qwen3.5
cat > ~/llm-servers/start-qwen3.sh << SCRIPT
#!/bin/bash
exec /Users/$(whoami)/llama.cpp/build/bin/llama-server \
  -m /Users/$(whoami)/llm-models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -c 65536 --host 127.0.0.1 --port 8081 \
  --flash-attn on --jinja -rea off \
  --temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --no-webui
SCRIPT

# Script Gemma4
cat > ~/llm-servers/start-gemma4.sh << SCRIPT
#!/bin/bash
exec /Users/$(whoami)/llama.cpp/build/bin/llama-server \
  -m /Users/$(whoami)/llm-models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --mmproj /Users/$(whoami)/llm-models/gemma4-26b/mmproj-BF16.gguf \
  -ngl 99 -c 65536 --host 127.0.0.1 --port 8082 \
  --flash-attn on -ctk q8_0 -ctv q8_0 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 1.0 --top-p 0.95 --top-k 64 --no-webui
SCRIPT

chmod +x ~/llm-servers/start-qwen3.6.sh \
         ~/llm-servers/start-qwen3.sh \
         ~/llm-servers/start-gemma4.sh
```

Los plists de launchd para el Modo A siguen el mismo patrón que el de llama-swap,
apuntando a cada script en `ProgramArguments`.

---

## 13. Configurar oh-my-openagent

### 13.1 Instalar dependencias

> **OpenCode Desktop** — existe una app nativa para macOS que incluye el TUI
> con panel lateral de cambios en tiempo real. Si prefieres una interfaz visual,
> descárgala desde [opencode.ai](https://opencode.ai) antes de continuar.
> Esta guía cubre la instalación CLI (equivalente en funcionalidad).

```bash
# Node.js (requerido para bunx/npx)
brew install node
node --version   # v20.x.x o superior

# GitHub CLI — requerido por oh-my-opencode doctor
brew install gh
gh --version     # gh version 2.x.x

# OpenCode CLI
npm install -g opencode-ai
opencode --version
```

### 13.2 Instalar oh-my-openagent

```bash
bunx oh-my-opencode install --no-tui \
  --claude=no \
  --openai=no \
  --gemini=no \
  --copilot=no
```

### 13.3 Verificar la instalación

```bash
bunx oh-my-opencode doctor
# Debe mostrar: ✅ No issues found
# Si reporta "GitHub CLI missing": brew install gh  (ya instalado arriba)
```

### 13.4 Configurar opencode.json

Config global con provider local, permisos de seguridad, agente por defecto y privacidad:

- `permission` — permisos granulares: git/docker/python sin preguntar, `.env` protegido, `rm -rf` y `sudo` bloqueados
- `default_agent: "sisyphus"` — oh-my-openagent orquesta desde el inicio sin seleccionar manualmente
- `share: "disabled"` — las sesiones no se suben a ningún servidor externo
- `instructions: ["AGENTS.md"]` — opencode inyecta el archivo en cada sesión si existe en el proyecto
- `model.limit` — declara contexto y output máximo por modelo para que opencode no desborde lo configurado en llama-swap
- `mcp.context7` — documentación bajo demanda: agrega `use context7` a cualquier prompt para obtener docs oficiales de LlamaIndex, Django, pgvector, etc.

```bash
cat > ~/.config/opencode/opencode.json << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["oh-my-openagent@latest"],

  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-swap Local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "qwen3.6-35b": {
          "name": "qwen3.6-35b",
          "limit": { "context": 98304, "output": 8192 }
        },
        "qwen3.5-9b": {
          "name": "qwen3.5-9b",
          "limit": { "context": 65536, "output": 4096 }
        },
        "gemma4-26b": {
          "name": "gemma4-26b",
          "limit": { "context": 65536, "output": 2048 }
        }
      }
    }
  },

  "default_agent": "sisyphus",

  "permission": {
    "bash": {
      "*":           "ask",
      "git *":       "allow",
      "docker *":    "allow",
      "python *":    "allow",
      "npm *":       "allow",
      "rm -rf *":    "deny",
      "sudo *":      "deny",
      "chmod 777 *": "deny"
    },
    "edit": "allow",
    "read": {
      "*":      "allow",
      ".env":   "deny",
      ".env.*": "deny"
    },
    "webfetch": "allow"
  },

  "mcp": {
    "context7": {
      "type": "remote",
      "url": "https://mcp.context7.com/mcp"
    },
    "playwright": {
      "type": "local",
      "command": ["npx", "-y", "@playwright/mcp@latest"]
    },
    "kratos-memory": {
      "type": "local",
      "command": ["npx", "kratos-mcp"]
    }
  },

  "instructions": ["AGENTS.md"],

  "share": "disabled"
}
EOF
```

> **`model.limit`** — `context` debe coincidir con el valor de `-c` en el `cmd` de
> `config.yaml`. `output` debe ser igual o menor al `max_tokens` configurado en
> `filters.setParams`. Si opencode intenta enviar más tokens de los que el servidor
> tiene asignados, la petición fallará silenciosamente.

> **Playwright MCP** (`@playwright/mcp`) — browser automation completo como herramienta MCP.
> Dos modos según el agente que lo invoque:
> - **Accessibility snapshot** (modo por defecto) — el agente recibe el árbol de accesibilidad
>   de la página como texto. Funciona con cualquier modelo (Qwen3.6, Qwen3.5). Bajo consumo de tokens.
> - **Visual screenshot** — el agente recibe una imagen de la página. Requiere modelo con visión:
>   usa `multimodal-looker` (gemma4-26b) para tareas que requieran ver el UI real.
>
> Instala el runner al primer uso — no hay nada que instalar manualmente.

> **kratos-memory MCP** (`kratos-mcp`) — memoria persistente ultra-lean por proyecto.
> Usa SQLite local para que los agentes recuerden contexto entre sesiones:
> nombres de variables, decisiones de arquitectura, errores previos.
> 100% project isolation: cada directorio tiene su propia base de datos.
> Sin servidores externos, sin configuración adicional.

> **Context7 MCP** — agrega `use context7` a cualquier prompt para que el agente
> busque la documentación oficial actualizada de la librería en cuestión:
> ```
> use context7 — cómo configurar VectorStoreIndex en LlamaIndex 0.12
> use context7 — parámetros de pgvector para búsqueda por similitud en Django
> ```
> No bloquea el contexto con toda la doc de antemano — la carga solo cuando la necesita.

> **`AGENTS.md`** — si creas este archivo en la raíz de tu proyecto, opencode lo
> inyecta automáticamente en cada sesión como contexto persistente. Útil para
> documentar el stack, convenciones y restricciones del proyecto:
>
> ```markdown
> # AGENTS.md
> ## Stack
> Django 4.2 + LlamaIndex + PostgreSQL (pgvector) + Redis
>
> ## Convenciones
> - Código en español donde sea posible
> - Tests con pytest, no unittest
> - Migraciones siempre con descripción explícita
>
> ## Restricciones
> - Nunca modificar archivos en /migrations/ directamente
> - Variables de entorno en .env.local, nunca hardcodeadas
> ```

> **`.rules/*.md`** — alternativa más granular a `AGENTS.md`. opencode aplica
> Rules con tres niveles de jerarquía en orden: **global** (`~/.config/opencode/rules/`)
> → **proyecto** (`.rules/` en la raíz) → **sub-proyecto** (`.rules/` en subdirectorios).
> Cada nivel sobreescribe o complementa el anterior. Útil para tener reglas globales
> de estilo aplicables a todos los proyectos, y restricciones específicas por repo:
>
> ```
> .rules/
>   stack.md          # "Usa Django + pgvector. Tests con pytest."
>   migrations.md     # "Nunca modifiques migraciones existentes directamente."
> ```
>
> `AGENTS.md` (vía `"instructions"`) y `.rules/` son compatibles — opencode carga ambos.
> Usa `AGENTS.md` para contexto de proyecto, `.rules/` para reglas de comportamiento reutilizables.

> Si usas el Modo A (servidores en paralelo), necesitas tres providers distintos
> apuntando a los puertos 8080, 8081 y 8082 respectivamente.

### 13.5 Configurar oh-my-openagent.json

Config completo optimizado para modelos locales con llama-swap:

- `categories` — routing automático por tipo de tarea al modelo correcto
- `fallback_models` — si Qwen3.6-35B falla, cae a Qwen3.5-9B
- `background_task.providerConcurrency` — limita a 1 porque llama-swap carga un modelo a la vez
- `runtime_fallback.timeout_seconds: 120` — llama-swap puede tardar 30s en cargar un modelo; 30s sería insuficiente
- `hephaestus` deshabilitado — diseñado exclusivamente para GPT-5.3-codex, no tiene prompt Claude/local
- `momus` deshabilitado — verificador de alta precisión diseñado para GPT, degradaría con modelos locales
- `prometheus.prompt_append` — guía a Prometheus para lanzar sub-agentes en paralelo; hereda el modelo de sisyphus
- `experimental.task_system: false` — desactivado; cuando estaba en `true` causaba bucles infinitos de orquestación entre agentes con modelos locales

```bash
cat > ~/.config/opencode/oh-my-openagent.json << 'EOF'
{
  "$schema": "https://raw.githubusercontent.com/code-yeongyu/oh-my-openagent/dev/assets/oh-my-opencode.schema.json",

  "agents": {
    "sisyphus": {
      "model": "local/qwen3.6-35b",
      "ultrawork": { "model": "local/qwen3.6-35b" },
      "fallback_models": ["local/qwen3.5-9b"]
    },
    "oracle": { "model": "local/qwen3.6-35b" },
    "prometheus": {
      "model": "local/qwen3.6-35b",
      "prompt_append": "Leverage deep & quick agents heavily, always in parallel."
    },
    "atlas": { "model": "local/qwen3.6-35b" },
    "explore": {
      "model": "local/qwen3.5-9b"
    },
    "librarian": {
      "model": "local/qwen3.5-9b"
    },
    "multimodal-looker": {
      "model": "local/gemma4-26b"
    },
    "hephaestus": { "disable": true },
    "momus":      { "disable": true }
  },

  "categories": {
    "quick":              { "model": "local/qwen3.5-9b" },
    "unspecified-low":    { "model": "local/qwen3.5-9b" },
    "unspecified-high":   { "model": "local/qwen3.6-35b" },
    "ultrabrain":         { "model": "local/qwen3.6-35b" },
    "visual-engineering": { "model": "local/gemma4-26b" },
    "writing":            { "model": "local/gemma4-26b" }
  },

  "background_task": {
    "defaultConcurrency": 1,
    "providerConcurrency": {
      "local": 1
    }
  },

  "runtime_fallback": {
    "enabled": true,
    "retry_on_errors": [429, 503, 529],
    "max_fallback_attempts": 2,
    "timeout_seconds": 120
  },

  "experimental": {
    "task_system": false
  },

  "tmux": { "enabled": false },
  "hashline_edit": false
}
EOF
```

### 13.6 Activar los agentes locales en cada proyecto (paso crítico)

> ℹ️ **Comportamiento de oh-my-openagent v3.17.4:** el plugin prioriza `oh-my-openagent.json`
> (nombre canónico) sobre `oh-my-opencode.json` (legacy). El instalador crea `oh-my-openagent.json`
> con defaults de `gpt-5-nano` — si no se sobreescribe, todos los agentes usan esos defaults.
> El config global en `~/.config/opencode/oh-my-openagent.json` **sí** tiene alcance global
> siempre que no exista un `oh-my-openagent.json` en `.opencode/` del proyecto que lo overridee.

**Symlink de proyecto (override por proyecto, opcional):**

```bash
# Crear symlink desde el config global al proyecto actual
ln -s ~/.config/opencode/oh-my-openagent.json \
      .opencode/oh-my-openagent.json

# Verificar
ls -la .opencode/oh-my-openagent.json
# .opencode/oh-my-openagent.json -> /Users/user/.config/opencode/oh-my-openagent.json
```

**Fix para todos los proyectos existentes de una vez:**

```bash
find ~/Documents -name ".opencode" -type d 2>/dev/null | while read dir; do
  if [ ! -f "$dir/oh-my-openagent.json" ]; then
    ln -s ~/.config/opencode/oh-my-openagent.json "$dir/oh-my-openagent.json"
    echo "✅ Symlink creado: $dir"
  else
    echo "⚠️  Ya existe: $dir/oh-my-openagent.json"
  fi
done
```

**Fix permanente para proyectos futuros — alias `omo-init`:**

Agrega esto a `~/.zshrc`. El alias crea el symlink de `oh-my-openagent.json` y también
inicializa el `opencode.json` del proyecto con el provider local si no existe:

```bash
cat >> ~/.zshrc << 'EOF'

# Inicializar oh-my-openagent en un proyecto nuevo
omo-init() {
  mkdir -p .opencode

  # Symlink de oh-my-openagent.json
  if [ ! -f ".opencode/oh-my-openagent.json" ]; then
    ln -s ~/.config/opencode/oh-my-openagent.json .opencode/oh-my-openagent.json
    echo "✅ oh-my-openagent.json symlink creado"
  else
    echo "ℹ️  oh-my-openagent.json ya existe"
  fi

  # opencode.json del proyecto si no existe
  if [ ! -f ".opencode/opencode.json" ]; then
    cp ~/.config/opencode/opencode.json .opencode/opencode.json
    echo "✅ opencode.json copiado al proyecto"
  else
    echo "ℹ️  opencode.json ya existe"
  fi

  echo "🚀 Listo — ejecuta: opencode"
}
EOF

source ~/.zshrc
```

Flujo completo para cualquier proyecto nuevo:

```bash
cd /nuevo/proyecto
omo-init    # ← crea symlink + copia opencode.json con provider local
opencode    # ← todos los agentes usan modelos locales
```

### 13.7 Uso básico

**Build Mode vs Plan Mode** — los dos modos fundamentales de operación:

| Modo | Cómo activar | Comportamiento |
|------|-------------|----------------|
| **Build Mode** | Por defecto (sisyphus) | El agente actúa directamente: edita archivos, corre comandos |
| **Plan Mode** | Tab → Prometheus | El agente propone un plan detallado para revisión humana antes de actuar |

Usa Plan Mode para tareas complejas o destructivas — es la revisión de seguridad antes de que el agente toque código real.

```bash
# Navegar al proyecto (con symlink ya creado via omo-init)
cd /ruta/a/tu/proyecto

# Abrir opencode
opencode

# Modo automático completo (Build Mode) — el agente orquesta todo
ultrawork implementa autenticación JWT siguiendo los patrones del proyecto

# Alias corto equivalente
ulw fix the failing tests

# Ejecutar sin abrir el TUI (headless — útil para CI/scripts)
opencode run "Explica la arquitectura del módulo de pagos"

# Continuar la última sesión
opencode run --continue "¿Qué más deberías mejorar?"

# Override de modelo para la sesión (sin tocar opencode.json)
opencode --model local/qwen3.5-9b   # forzar el modelo rápido para esta sesión
opencode --model local/gemma4-26b   # forzar multimodal para esta sesión

# Modo planificación precisa (Plan Mode)
# Presiona Tab → selecciona Prometheus → describe la tarea
# Responde las preguntas del agente → ejecuta /start-work

# LSP — diagnósticos reales del compilador inyectados automáticamente
# Si tienes un language server en $PATH (pyright, typescript-language-server,
# gopls, rust-analyzer...) opencode lo detecta y lo usa sin configuración.
# El agente verá errores reales con archivo + línea en vez de adivinar:
#   [LSP] src/auth.py:42 — Cannot access attribute "user" on "None"
# Para deshabilitar: "lsp": { "disabled": true } en opencode.json

# Documentación bajo demanda con Context7 MCP
# Agrega "use context7" a cualquier prompt:
use context7 — cómo configurar VectorStoreIndex con pgvector en LlamaIndex 0.12
use context7 — parámetros de similarity_search en Django pgvector

# BMAD personas como Skills de OpenCode
# BMAD-METHOD (github.com/bmad-code-org/BMAD-METHOD) define roles de equipo
# (PM, Architect, Developer, UX) como archivos Markdown puros.
# Puedes usarlos directamente como Skills en opencode:
#
# .opencode/skills/
#   pm-persona.md        # Product Manager — requirements, user stories
#   architect-persona.md # System Architect — design decisions, ADRs
#   dev-persona.md       # Developer — implementation patterns, conventions
#
# En el prompt: "acting as architect, review the auth module design"
# El agente carga el persona file como contexto adicional.
```

> **Nota para RTX 5090 (Linux):** En el servidor Linux con CUDA puedes reemplazar
> `-ctk q8_0 -ctv q8_0` por `-ctk turbo3 -ctv turbo3` usando el fork TurboQuant
> de llama.cpp (`github.com/spiritbuun/llama-cpp-turboquant-cuda`). Ofrece 3.5x
> más compresión del KV cache que q8_0, con perplexity que bate q8_0 baseline.
> En 131K contexto ahorra ~1,830 MiB de VRAM vs f16 estándar.
> **No aplica a Apple Silicon** — requiere CUDA, no Metal.

---

## 13. Verificación final

```bash
cat > ~/llm-servers/check-status.sh << 'ENDSCRIPT'
#!/bin/bash
echo "======================================"
echo "  Verificación de Setup llama.cpp"
echo "======================================"

echo ""
echo "1. Binarios..."
command -v llama-server &>/dev/null \
  && echo "   ✅ llama-server: $(which llama-server)" \
  || echo "   ❌ llama-server NO encontrado"
command -v llama-swap &>/dev/null \
  && echo "   ✅ llama-swap: $(which llama-swap)" \
  || echo "   ❌ llama-swap NO encontrado"

echo ""
echo "2. Modelos en disco..."
for model in \
  "$HOME/llm-models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  "$HOME/llm-models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf" \
  "$HOME/llm-models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf" \
  "$HOME/llm-models/gemma4-26b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
  "$HOME/llm-models/gemma4-26b/mmproj-BF16.gguf"
do
  if [ -f "$model" ]; then
    size=$(du -sh "$model" | cut -f1)
    echo "   ✅ $(basename $model) ($size)"
  else
    echo "   ❌ NO encontrado: $model"
  fi
done

echo ""
echo "3. Endpoint llama-swap activo..."
response=$(curl -s --max-time 5 "http://127.0.0.1:8080/health" 2>/dev/null)
if echo "$response" | grep -qi "ok"; then
  echo "   ✅ llama-swap responde en :8080"
else
  echo "   ❌ llama-swap NO responde — verifica launchd"
fi

echo ""
echo "4. Modelo activo en RAM..."
curl -s http://127.0.0.1:8080/running 2>/dev/null | \
  python3.11 -c "
import json,sys
try:
  d=json.load(sys.stdin)
  running=d.get('running',[])
  if running:
    for m in running: print('   🟢 Activo:', m['model'], '— estado:', m.get('state','?'))
  else:
    print('   ℹ️  Ningún modelo cargado (se cargan bajo demanda)')
except: print('   ⚠️  No se pudo leer /running')
" 2>/dev/null

echo ""
echo "5. Test de inferencia (qwen3.5-9b — puede tardar 30s)..."
result=$(curl -s --max-time 90 http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Di OK"}],"max_tokens":50}' \
  2>/dev/null)
if echo "$result" | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
print('   ✅ Respuesta:', msg['content'].strip())
if msg.get('reasoning_content'):
  print('   ⚠️  reasoning_content presente (no esperado para qwen3.5-9b)')
" 2>/dev/null; then
  true
else
  echo "   ❌ Fallo en inferencia"
fi

echo ""
echo "======================================"
echo "  Procesos activos"
echo "======================================"
ps aux | grep -E "llama-swap|llama-server" | grep -v grep | \
  awk '{printf "   PID %s (%s): %.0f MB RAM\n", $2, $11, $6/1024}'
ENDSCRIPT

chmod +x ~/llm-servers/check-status.sh
~/llm-servers/check-status.sh
```

---

### 13.8 MCPs disponibles — estado, instalación y autenticación

| MCP | Estado típico | Descripción |
|-----|--------------|-------------|
| `context7` | ✅ Connected | Docs oficiales bajo demanda — `use context7` en el prompt |
| `playwright` | ✅ Connected | Browser automation (accesibilidad + visual con gemma4-26b) |
| `kratos-memory` | ✅ Connected | Memoria SQLite por proyecto entre sesiones |
| `context-mode` | ✅ Connected | Protege el context window — sandbox para comandos largos |
| `mcp-toolkit:mcp-gateway` | ✅ Connected | Browser + herramientas dev adicionales |
| `grep_app` | ✅ Connected | Búsqueda de código en GitHub vía grep.app |
| `figma:figma` | ⚠️ Needs auth | Implementar diseños desde Figma directamente |
| `postman:postman` | ⚠️ Needs auth | Testing de APIs y colecciones Postman |
| `grepai` | ❌ Error -32000 | Búsqueda semántica — requiere fix de instalación |
| `websearch` | ❓ Sin estado | Búsqueda web — requiere configuración |

#### MCPs sin requisitos adicionales (funcionan con `npx`)

Estos ya están en `opencode.json` o se autoinstalan:

```bash
# Verificar que Node.js está disponible (requerido para npx)
node --version    # v20+

# context7 — remote, no requiere instalación local
# playwright — se descarga al primer uso con npx -y
# kratos-memory — requiere instalación previa:
npm install -g kratos-mcp
```

#### Figma MCP — autenticación

```bash
# 1. Obtener token de acceso personal en:
#    figma.com → Settings → Security → Personal access tokens

# 2. Añadir al MCP en opencode.json:
#    "figma": {
#      "type": "remote",
#      "url": "https://mcp.figma.com/mcp",
#      "headers": { "Authorization": "Bearer TU_FIGMA_TOKEN" }
#    }

# O en Claude Code vía variable de entorno:
echo 'export FIGMA_API_KEY="figd_xxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

#### Postman MCP — autenticación

```bash
# 1. Obtener API key en:
#    go.postman.co → Settings → API Keys → Generate API Key

# 2. Añadir al MCP en opencode.json:
#    "postman": {
#      "type": "local",
#      "command": ["npx", "-y", "@postman/mcp-server"],
#      "env": { "POSTMAN_API_KEY": "PMAK-xxxxxxxxxxxx" }
#    }
```

#### grepai — fix error -32000 (Connection closed)

Error -32000 significa que el proceso inicia y se cierra inmediatamente — casi siempre falta el binario o dependencia:

```bash
# Verificar si grepai-mcp está instalado
which grepai-mcp 2>/dev/null || echo "no encontrado"

# Instalarlo (requiere Go o npx según la versión)
npm install -g grepai-mcp 2>/dev/null || \
  go install github.com/nicholasgasior/grepai-mcp@latest 2>/dev/null

# Si usa el binario standalone de Claude Code superpowers:
# Verificar que existe en la ruta configurada
ls ~/.claude/plugins/cache/*/grepai*/bin/ 2>/dev/null

# Verificar la configuración MCP en Claude Code settings
cat ~/.claude/settings.json | grep -A5 grepai
```

#### websearch — configuración

```bash
# En opencode.json, agregar bajo "mcp":
#    "websearch": {
#      "type": "local",
#      "command": ["npx", "-y", "@opencode-ai/websearch-mcp"]
#    }

# O la variante con Tavily (mejor calidad):
#    "websearch": {
#      "type": "local",
#      "command": ["npx", "-y", "tavily-mcp"],
#      "env": { "TAVILY_API_KEY": "tvly-xxxxxxxxxxxx" }
#    }
# API key gratuita en: tavily.com
```

#### Verificar estado de todos los MCPs en opencode

```bash
# Dentro del TUI de opencode: presiona Ctrl+M para ver la lista de MCPs activos
# O desde la CLI:
opencode run "list all available mcp tools" --no-stream 2>/dev/null | head -20
```

---

## 14. Referencia rápida de comandos

```bash
# ─── VER ESTADO ──────────────────────────────────────────────
launchctl list | grep llamaswap              # servicio activo
curl -s http://127.0.0.1:8080/health         # health check → "OK"
curl -s http://127.0.0.1:8080/v1/models      # modelos registrados (con aliases)
curl -s http://127.0.0.1:8080/running        # modelo ACTIVO en RAM ahora mismo
open http://127.0.0.1:8080                   # UI web de llama-swap

# ─── CONTROL DE MODELOS EN RAM ───────────────────────────────
curl -s http://127.0.0.1:8080/unload         # descargar todo de RAM

# ─── ACCESO DIRECTO AL UPSTREAM ──────────────────────────────
curl -s http://127.0.0.1:8080/upstream/qwen3.5-9b/health
curl -s http://127.0.0.1:8080/upstream/qwen3.5-9b/slots

# ─── LOGS EN TIEMPO REAL ─────────────────────────────────────
curl -sN http://127.0.0.1:8080/logs/stream                    # todos
curl -sN http://127.0.0.1:8080/logs/stream/proxy              # solo proxy
curl -sN http://127.0.0.1:8080/logs/stream/qwen3.6-35b    # un modelo
curl -sN 'http://127.0.0.1:8080/logs/stream?no-history'       # solo nuevos
tail -f ~/llm-servers/llamaswap-err.log                        # errores launchd

# ─── CONTROL DEL SERVICIO ────────────────────────────────────
launchctl bootout  gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist  # detener
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist # iniciar
launchctl kickstart -k gui/$(id -u)/com.llmserver.llamaswap               # restart rápido
launchctl print gui/$(id -u)/com.llmserver.llamaswap                     # debug detallado

# ─── EDITAR CONFIG SIN REINICIAR ─────────────────────────────
nano ~/llm-servers/config.yaml
# --watch-config detecta el cambio y recarga automáticamente

# ─── TEST RÁPIDO DE CADA MODELO ──────────────────────────────
for M in qwen3.6-35b qwen3.5-9b gemma4-26b; do  # qwen3.6-35b = modelo principal
  echo "→ $M"
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Di OK\"}],\"max_tokens\":50}" \
    | python3.11 -c "
import json,sys
d=json.load(sys.stdin)
msg=d['choices'][0]['message']
print('  content:', msg['content'][:60])
print('  reasoning:', bool(msg.get('reasoning_content','')))
"
done

# ─── TEST DE ALIAS ────────────────────────────────────────────
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Di OK"}],"max_tokens":50}' \
  | python3.11 -c "import json,sys; d=json.load(sys.stdin); print(d['model'], d['choices'][0]['message']['content'])"
# Debe responder como qwen3.6-35b (alias configurado)

# ─── MONITOREO DE TOKENS POR SEGUNDO ─────────────────────────
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","stream":true,"timings_per_token":true,
       "messages":[{"role":"user","content":"Escribe 3 oraciones."}]}' \
  | grep -o '"predicted_per_second":[0-9.]*'

# ─── ACTUALIZAR llama.cpp ─────────────────────────────────────
cd ~/llama.cpp && git pull
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
launchctl kickstart -k gui/$(id -u)/com.llmserver.llamaswap

# ─── ACTUALIZAR llama-swap ────────────────────────────────────
brew upgrade llama-swap
launchctl kickstart -k gui/$(id -u)/com.llmserver.llamaswap

# ─── MEMORIA EN USO ──────────────────────────────────────────
ps aux | grep -E "llama-swap|llama-server" | grep -v grep

# ─── USAR opencode ────────────────────────────────────────────
cd /tu/proyecto
opencode
# Escribe 'ultrawork' en el prompt para activar Sisyphus
# Presiona Tab para entrar en modo Prometheus (planificador)
```

---

## 15. Solución de problemas frecuentes

### ❌ Error: "Metal GPU not found" o "using CPU"

> **Impacto real:** Sin Metal obtienes **5–15 t/s** (CPU puro).
> Con Metal en M4: **40–60 t/s**. Es una diferencia de 4–8x — vale la pena resolver esto antes de continuar.

```bash
llama-server --version 2>&1 | grep -i metal
# Debe mostrar: GGML_USE_METAL=1

# Confirmar que las capas se están offloadeando a la GPU:
# En los logs de arranque busca:
# llm_load_tensors: offloaded 64/64 layers to GPU
# Si ves "offloaded 0/64" o ausencia de Metal, recompila:

cd ~/llama.cpp && rm -rf build
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### ❌ El modelo genera thinking (respuesta vacía o [Start thinking])

Cada modelo tiene su propio mecanismo de control:

```bash
# Qwen3.6-35B: el thinking es controlado por el chat template.
# Asegúrate de tener --jinja en el cmd. El chat template embebido en el GGUF
# decide cuándo activar el reasoning. Comportamiento normal en llama-cli.

# Qwen3.5-9B: thinking desactivado por defecto en modelos Small.
# Si aparece, agrega al cmd: -rea off
# Nota: -rea off solo funciona en llama-server, no en llama-cli.

# Gemma 4: método canónico del chat template.
# Agrega al cmd: --jinja --chat-template-kwargs '{"enable_thinking":false}'
# Nota: --chat-template-kwargs NO funciona en llama-cli, solo en llama-server.

# Verificar flags disponibles en tu build:
llama-server --help 2>&1 | grep -i "reasoning\|think"
```

### ❌ Respuesta vacía (content: "")

```bash
# El max_tokens se agota dentro del bloque de thinking.
# Solución: aumentar max_tokens o usar filters en config.yaml:
#
# filters:
#   setParams:
#     max_tokens: 8192   # Qwen3.6-35B
#     max_tokens: 2048   # Gemma 4
#
# Mínimo recomendado para peticiones manuales: 500 tokens
```

### ❌ Error 502 en llama-swap

```bash
# 502 con tiempo < 1ms = llama-server no llegó a iniciar.

# 1. Usar ruta absoluta del binario en config.yaml
which llama-server
# Úsala en el cmd: /Users/tu-usuario/llama.cpp/build/bin/llama-server

# 2. Verificar nombre exacto del archivo GGUF
ls ~/llm-models/qwen3.6-35b/

# 3. Ver el error exacto corriendo en primer plano:
llama-swap --config ~/llm-servers/config.yaml --listen 127.0.0.1:8080

# 4. Seguir log del modelo mientras intenta arrancar:
curl -sN http://127.0.0.1:8080/logs/stream/qwen3.5-9b
```

### ❌ Gemma 4 — freeze o timeout con Flash Attention en Apple Silicon

```bash
# Agregar al cmd en config.yaml:
# -ctk q8_0 -ctv q8_0
# (cache type q8_0 para key y value del KV cache)
```

### ❌ llama-swap no arranca / no responde en :8080

```bash
# 1. Ver log de errores
tail -50 ~/llm-servers/llamaswap-err.log

# 2. Debug detallado del servicio (PID, exit code, variables de entorno inyectadas)
launchctl print gui/$(id -u)/com.llmserver.llamaswap

# 3. Correr en primer plano
llama-swap --config ~/llm-servers/config.yaml --listen 127.0.0.1:8080

# 4. Verificar rutas en config.yaml
grep "llm-models" ~/llm-servers/config.yaml

# 5. Validar sintaxis YAML
python3.11 -c "import yaml; yaml.safe_load(open('$HOME/llm-servers/config.yaml'))" \
  && echo "YAML OK" || echo "Error de sintaxis YAML"

# 6. Si load/unload da warning, usar la API moderna:
launchctl bootout  gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist
```

### ❌ Modelo tarda mucho / se queda en "loading"

Comportamiento normal la primera vez. Tiempos esperados en M4:

| Modelo | Tiempo de carga inicial |
|---|---|
| Qwen3.5-9B (~5 GB) | ~5–10 s |
| Gemma 4 26B (~16 GB) | ~15–25 s |
| Qwen3.6-35B-A3B (~22 GB) | ~20–30 s |

```bash
# Monitorear estado
curl -s http://127.0.0.1:8080/running | python3.11 -m json.tool
# state: "loading" → "ready"

# Seguir log mientras carga
curl -sN http://127.0.0.1:8080/logs/stream/qwen3.6-35b

# Si supera 120s, aumentar en config.yaml:
# healthCheckTimeout: 240
```

### ❌ Intentas correr llama-server dentro de Docker

**Metal no funciona dentro de contenedores en macOS.** Los contenedores Docker corren
en una VM Linux — esa VM no tiene drivers Metal. Obtendrás CPU-only (5–15 t/s) en vez
de los 40–60 t/s de Metal nativo.

```bash
# Diagnóstico: si ves esto en los logs, estás en CPU-only
# ggml_metal_init: Metal is not supported on this device
# llm_load_tensors: offloaded 0/64 layers to GPU

# Solución: llama-server SIEMPRE debe correr como proceso nativo macOS.
# llama-swap (Modo B) ya garantiza esto — gestiona llama-server como proceso nativo.
# No hay workaround para Metal dentro de contenedores: es una limitación de macOS.
```

### ❌ Puerto 8080 ya en uso

```bash
lsof -i :8080
kill -9 <PID>
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.llmserver.llamaswap.plist
```

### ❌ Descarga interrumpida o timeout

```bash
# hf download retoma automáticamente — re-ejecuta el mismo comando
hf download \
  unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --local-dir ~/llm-models/qwen3.6-35b

# Si la descarga se interrumpe por timeout en conexiones lentas,
# aumenta el tiempo de espera (default: 10 segundos):
export HF_HUB_DOWNLOAD_TIMEOUT=120
hf download ...   # reintentar con el mismo comando

# Para ver qué archivos faltan antes de reintentar:
hf download unsloth/Qwen3.6-35B-A3B-GGUF --dry-run
```

### ❌ opencode no conecta con los modelos

```bash
# 1. Confirmar que llama-swap está corriendo
curl -s http://127.0.0.1:8080/health

# 2. Verificar baseURL en opencode.json
cat ~/.config/opencode/opencode.json | grep baseURL

# 3. Test manual con curl
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"test"}],"max_tokens":50}'

# 4. Revisar log de errores
tail -50 ~/llm-servers/llamaswap-err.log
```

### ❌ Los agentes (oracle, prometheus, explore...) usan Zen/Anthropic en vez de modelos locales

oh-my-openagent v3.17.4 tiene dos configs: `oh-my-openagent.json` (canónico, prioridad) y
`oh-my-opencode.json` (legacy). El instalador crea `oh-my-openagent.json` con `gpt-5-nano`.
Si `~/.config/opencode/oh-my-openagent.json` no tiene tus modelos locales, todos los agentes
caen a los defaults de Anthropic.

```bash
# Diagnóstico — ver qué modelo usa sisyphus
grep -A2 '"sisyphus"' ~/.config/opencode/oh-my-openagent.json
# Debe mostrar: "model": "local/qwen3.6-35b"
# Si muestra "opencode/gpt-5-nano" → el instalador sobreescribió tu config

# Fix — restaurar config con modelos locales
cat > ~/.config/opencode/oh-my-openagent.json << 'OMOEOF'
# (pegar contenido de la sección 13.5)
OMOEOF

# Fix para todos los proyectos existentes (symlinks)
find ~/Documents -name ".opencode" -type d 2>/dev/null | while read dir; do
  if [ ! -f "$dir/oh-my-openagent.json" ]; then
    ln -s ~/.config/opencode/oh-my-openagent.json "$dir/oh-my-openagent.json"
    echo "✅ $dir"
  fi
done

# Fix permanente para proyectos futuros
# Asegúrate de tener el alias omo-init en ~/.zshrc (ver sección 13.6)
# y ejecuta siempre omo-init al empezar en un proyecto nuevo
omo-init
```

---

*Guía generada para Mac Studio M4 64GB — Abril 2026*
*Stack: llama.cpp + llama-swap + Qwen3.6-35B-A3B · Qwen3.5-9B · Gemma-4-26B-A4B*
*Fuentes: [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5) · [Unsloth Gemma 4](https://unsloth.ai/docs/models/gemma-4) · [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) · [context7/llama.cpp](https://context7.com/ggml-org/llama.cpp/llms.txt) · [context7/llama-swap](https://context7.com/mostlygeek/llama-swap/llms.txt) · [context7/oh-my-openagent](https://context7.com/code-yeongyu/oh-my-openagent/llms.txt) · [context7/opencode](https://context7.com/anomalyco/opencode/llms.txt) · [jamesarslan/local-ai-coding-setup](https://github.com/jamesarslan/local-ai-coding-setup)*
