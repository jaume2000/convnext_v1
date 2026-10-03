# Informe de experimentos — ¿Son las redes residuales discretizaciones de una ODE?

Fecha: 3 oct 2026. Fuentes: CSVs en `outputs/` (interpolación, ablation shared, historiales de entrenamiento, `featureMaps*`, `zero_outlier_channel_shared`).

---

## 0. Resumen ejecutivo

| # | Pregunta | Respuesta corta |
|---|---|---|
| 1 | ¿Aguantan ConvNeXt / Swin / ResNet más pasos de Euler en el stage 3 (horizonte fijo)? | **Sí.** Con pesos "planos" (repetir cada bloque R veces con ES=1/R) todas convergen a un límite continuo que pierde solo 0.6–0.9 pts (2.3 en ConvNeXt sin droppath). Si no se conserva el horizonte (ES=1 con R>1) colapsan a ~0%. |
| 2 | ¿Se puede entrenar un stage 3 con un único bloque compartido? | **Sí.** 80.34% vs 80.96% (−0.6 pts). Es la red más "ODE": converge a 80.12% con cualquier solver/D y es robusta a D si T=D·ES≈9. |
| 3 | ¿Qué dicen las métricas de dinámica? | Con todos los canales, todas las ConvNeXt parecen trayectorias casi rectas y 1-D (R≈0.94–0.98, PR≈1.1). **Es un artefacto de un único canal masivo.** |
| 4 | ¿El droppath es lo que hace a ConvNeXt "Euler-compatible"? | **Parcialmente.** Sin droppath el gap al límite ODE pasa de −0.73 a −2.31 pts y bilinear se hunde (−21.8). Pero ResNet y Swin también son compatibles, así que no es condición necesaria. |
| 5 | Massive activations | ConvNeXt (las 3 variantes) tiene **un canal que concentra 98–99% de la energía de x** a la salida del stage 3 (ch 236 shared: ‖x‖=3570 vs 140 el siguiente). Swin tiene uno moderado (~17%), ResNet ninguno (<1%). Sin él la dinámica es rotacional, curva y multidimensional. |
| 6 | ¿Qué hace ese canal? | **No retroalimenta la dinámica del resto de canales** (borrarlo en la ODE ≡ ignorarlo al medir, <0.5% diferencia), pero su borrado cuesta **−12.5 pts** (80.34→67.81). Su efecto es *downstream* (stage 4 / normalización). La hipótesis "cronómetro interno" queda descartada dentro del stage 3; "cronómetro/escala leída por el stage 4" sigue abierta. |
| 7 | Euler vs RK2 vs RK4 | Las redes están **afinadas a Euler ES=1**: RK2/RK4 en el paso de entrenamiento pierden 0.15–2.5 pts y dan directamente el límite ODE. Con pasos grandes (D pequeño, T fijo) RK4 ≫ Euler. |
| 8 | ¿Interpolación bilinear de pesos? | **No funciona en ningún modelo** (−7 a −24 pts a R=128). Solo la interpolación "plana" (constante a trozos) preserva la accuracy. |

---

## 1. Notación

- **Stage 3**: el stage largo de la red. Bloques residuales interpolados: ConvNeXt-T 9, Swin-T 6, ResNet-50 5, ResNet-101 22 (el bloque con downsample queda fuera).
- **Bloque residual como paso de Euler**: `x ← x + ES · h`, con `h = block(x) − x`.
- **R** (interpolados): cada bloque nativo se aplica R veces con paso **ES**. Horizonte total por bloque = R·ES.
- **D / ES** (shared): un único bloque aplicado D veces; horizonte **T = D·ES** (entrenamiento: D=9, ES=1, T=9).
- **plain**: pesos constantes a trozos (bloque k durante sus R micro-pasos). **bilinear**: pesos interpolados linealmente entre bloque k y k+1.
- Métricas de dinámica (`feature_map_explorer.py`): ‖h‖, ‖x‖, ω (velocidad angular de h), aceleración y su descomposición tangencial/normal, curvatura κ, α, rectitud R=N/L, PR (participation ratio) en profundidad y espacial, distancia a la trayectoria R1 nativa.

---

## 2. Experimento 1 — Más pasos de Euler en redes preentrenadas

ImageNet val (50k), top-1 %. Interpolación **plain**, **Euler (RK1)**, horizonte conservado (ES=1/R).

| R | ES | ConvNeXt-T | ConvNeXt-T dp0 | Swin-T | ResNet-50 | ResNet-101 |
|---|---|---|---|---|---|---|
| 1 | 1 | **80.96** | **79.27** | **81.08** | **80.34** | **81.67** |
| 2 | 0.5 | 80.78 | 78.77 | 80.90 | 80.06 | 81.58 |
| 4 | 0.25 | 80.52 | 78.05 | 80.65 | 79.84 | 81.38 |
| 10 | 0.1 | 80.36 | 77.51 | 80.47 | 79.61 | 81.22 |
| 32 | 0.031 | 80.29 | 77.15 | 80.38 | 79.50 | 81.12 |
| 128 | 0.0078 | 80.24 | 77.02 | 80.34 | 79.45 | 81.06 |
| **Límite ODE** (RK4) | — | 80.23 | 76.96 | 80.32 | 79.43 | 81.05 |
| **Gap R1 → ODE** | | **−0.73** | **−2.31** | **−0.76** | **−0.91** | **−0.62** |
| R=10, **ES=1** (T×10) | | 0.14 | 0.08 | 1.03 | 0.10 | 0.11 |
| R=100, **ES=1** (T×100) | | 0.12 | 0.12 | 0.10 | 0.10 | 0.10 |

Además: `convnext_r3_quick` (R=3, ES=1) → 41.03%.

**Análisis**
- Todas las arquitecturas residuales se comportan como discretizaciones de Euler de una ODE: refinando el paso, la accuracy **converge monótonamente** a un límite continuo bien definido (idéntico al que da RK4).
- El límite está siempre **por debajo** de la red original: la red explota el error de discretización de Euler ES=1 (≈0.6–0.9 pts; 2.3 en ConvNeXt dp0).
- Lo que importa es el **horizonte**: repetir bloques sin escalar el paso (ES=1) destruye la red.

---

## 3. Experimento 2 — Stage 3 compartido (un único bloque, D=9, ES=1)

### 3.1 Entrenamiento (300 épocas, misma receta)

| Modelo | Best val | Last val | Last train* |
|---|---|---|---|
| ConvNeXt-T (droppath 0.1) | 80.97 | 80.96 | 68.84 |
| ConvNeXt-T droppath 0 | 79.37 | 79.27 | 72.92 |
| **ConvNeXt-T shared stage 3** | **80.36** | **80.34** | 65.88 |
| Shared D27 (en curso, 33/300 ep) | 70.52 | 70.52 | 46.46 |
| Shared all-stages (en curso, 53/300 ep) | 69.95 | 69.95 | 47.33 |

\*train con augmentations/mixup, no comparable con val.

Compartir los 9 bloques del stage 3 (÷9 parámetros del stage) cuesta solo **−0.6 pts**.

### 3.2 Ablation de profundidad / paso (`sharedConvnextAblation/results.csv`)

**ES=1 fijo (el horizonte crece con D):**

| D | 1 | 3 | 5 | 7 | 8 | **9** | 10 | 12 | 18 | 24 | 32 | 48 | 128 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 33.6 | 72.1 | 78.9 | 80.1 | 80.3 | **80.34** | 80.3 | 80.1 | 78.5 | 73.4 | 49.1 | 3.2 | 0.3 | 0.06 |

**Horizonte fijo T=9 (ES=9/D):**

| D | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 12 | 32 | 128 | 1024 | 10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Euler | 47.1 | 5.9 | 37.5 | 67.8 | 77.2 | 79.5 | 80.2 | 80.4 | **80.34** | 80.35 | 80.22 | 80.14 | 80.12 | 80.12 |
| RK2 | 0.6 | 10.4 | 54.5 | 73.6 | 78.7 | 79.9 | 80.1 | 80.2 | 80.19 | 80.14 | 80.11 | 80.12 | 80.12 | — |
| RK4 | 2.7 | 48.0 | 71.4 | 78.9 | 80.0 | 80.1 | 80.1 | 80.1 | 80.11 | 80.12 | 80.12 | 80.12 | 80.12 | — |

**D=9 fijo, variando T (ES variable):**

| T = 9·ES | 0.09 | 0.9 | 1.1 | 2.25 | 3.6 | 4.5 | 7.4 | **9** | 16.2 | 18 | 32.4 | 36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top-1 | 11.8 | 35.5 | 43.6 | 67.7 | 76.7 | 78.7 | 80.2 | **80.34** | 76.3 | 71.9 | 4.6 | 2.3 |

> Nota: las antiguas filas "D=45, ES=9/x" eran en realidad **D=9** con ES=9/x (`block_indices = range(9)`). Corregido: `results.csv` usa ahora nombres canónicos `D{D}_ES{ES}_T{T}_{método}` con columna `T` (original en `results_legacy_labels.csv`). Como el barrido de T a D=9 mezcla el efecto del horizonte con el error de un paso grueso (ES hasta 3.6), se ha añadido la sección **D** (barrido fino de T con Euler ES=0.1 y RK4 ES=0.25), pendiente de ejecutar.

**Análisis**
- El bloque compartido define un **campo vectorial** cuya integral hasta **T≈9** es lo que la cabeza espera. La accuracy es una función casi exclusiva de **T**, no de D: con T=9 cualquier D≥7 vale (80.1–80.4), y el límite continuo es **80.12%** (−0.22 respecto a Euler ES=1).
- El gap Euler→ODE del shared (−0.22) es **3× menor** que el de ConvNeXt no compartido (−0.73): compartir pesos produce una ODE "más auténtica".
- Integrar de menos (T<9) o de más (T>9) degrada de forma suave al principio y colapsa lejos de T=9 ⇒ el horizonte se comporta como un **tiempo de integración aprendido**.

---

## 4. Experimento 3 — Métricas de dinámica y vídeos

Ejecutadas con `scripts/feature_map_explorer.py` (vídeos de h, x, ‖h‖, cos, scatter/spaghetti por canal, y `metrics/` con 32 paneles + CSVs). **Todas sobre una única imagen** (val #644, clase 289 snow leopard).

### 4.1 Con todos los canales (R100 ES0.01 plain para interpolados; shared D100 ES0.09)

| Run | Rectitud R | PR_depth | PR_spatial | κ medio (intra) | ω medio (intra) | a_n/‖a‖ | cos(x₀,x_T) |
|---|---|---|---|---|---|---|---|
| shared D9 ES1 | 0.967 | 1.10 | 1.03 | 0.0007 | 0.23 | 0.67 | 0.63 |
| shared D100 ES0.09 | 0.976 | 1.07 | 1.03 | 0.0005 | 0.15 | 0.58 | 0.64 |
| shared D100 RK4 | 0.976 | 1.07 | 1.03 | 0.0005 | 0.14 | 0.57 | 0.64 |
| ConvNeXt R100 | 0.941 | 1.17 | 1.14 | 0.0100 | 0.35 | 0.85 | 0.50 |
| ConvNeXt dp0 R100 | 0.932 | 1.05 | 7.42 | n/d | 0.57 | n/d | 0.35 |
| Swin R100 | 0.823 | 1.17 | 6.81 | 0.0042 | 0.52 | 0.91 | 0.08 |
| ResNet-50 R100 | 0.389 | 5.12 | 6.40 | 0.0024 | 0.67 | 0.90 | 0.57 |
| ResNet-101 R100 | 0.162 | 20.09 | 13.30 | 0.0022 | 0.53 | 0.92 | 0.46 |

n/d = run antiguo sin las columnas nuevas (hay que re-ejecutar con `--metrics-only`).

**Análisis**
- Hay un gradiente claro: **shared > ConvNeXt > Swin > ResNet** en rectitud y baja dimensionalidad. ResNet-101 es una trayectoria muy tortuosa (R=0.16, PR_depth=20).
- RK4 y Euler con paso fino dan trayectorias indistinguibles (shared D100), coherente con la convergencia en accuracy.
- Shared (todos los canales): `h` casi no gira (cos(h₀,h_T)=0.75); fase de "acelerar y luego frenar" (a_t/‖a‖ ≈ +0.87 → −0.90 a t≈4).
- ConvNeXt no compartido: dientes de sierra; dentro de cada bloque ‖h‖ decrece (contracción) y en cada cambio de bloque hay un salto enorme de h (aceleración 10²–10⁴× la intra-bloque).

---

## 5. Experimento 4 — Hipótesis del droppath

| | ConvNeXt (dp 0.1) | ConvNeXt (dp 0) | Δ |
|---|---|---|---|
| Val R1 (original) | 80.96 | 79.27 | −1.69 |
| Límite ODE (plain RK4) | 80.23 | 76.96 | −3.27 |
| **Gap R1 → ODE** | **−0.73** | **−2.31** | 3.2× mayor |
| Bilinear R128 | 73.66 | 57.44 | −16.2 |
| RK2 en ES=1 | 80.23 | 76.73 | |

**Análisis**
- El droppath (stochastic depth) obliga a que cada bloque sea "opcional" ⇒ actualizaciones pequeñas y coherentes con sus vecinos ⇒ la red se parece más a una ODE. Sin droppath el gap al límite continuo se triplica y la interpolación bilinear se hunde.
- **Pero no es la única causa**: ResNet-50/101 y Swin-T también convergen con gaps de 0.6–0.9. La propiedad "Euler" parece ser genérica de las redes residuales profundas; el droppath la **refuerza**.
- Pendiente: verificar la receta de los pesos torchvision (`weights="DEFAULT"`) de ResNet/Swin (¿usan stochastic depth?) y probar tasas intermedias (0.05, 0.2) o un shared con dp=0 para cerrar la causalidad.

---

## 6. Experimento 5 — Massive activations y dinámica sin ese canal

### 6.1 Detección

Canal ignorado = top-1 por ‖h‖ (W×H) en la trayectoria media. Fracción de energía del canal = 1 − (‖·‖ sin canal / ‖·‖ total)²:

| Modelo | Canal | x en t=0 | x en t=T | h en t=0 | h en t=T |
|---|---|---|---|---|---|
| ConvNeXt shared | 236 | 39% | **98%** | **96%** | 92% |
| ConvNeXt | 195 | 25% | **99%** | 5% | 96% |
| ConvNeXt dp0 | 225 | 12% | **99%** | 10% | 99% |
| Swin-T | 322 | 1% | 17% | 1% | 15% |
| ResNet-50 | 780 | 0% | 0% | 3% | 0% |
| ResNet-101 | 750 | 0.5% | 0.7% | 0% | 0% |

En el shared (R1, salida del stage 3): ‖x₂₃₆‖ = **3570**, el 2.º canal = 140 (25× más grande). En ResNet no hay massive activation: el `_ignore1` ahí es solo "quitar el canal más activo", sin efecto.

Diferencia importante: en el **shared el canal domina h desde t=0** (es lo que el campo empuja siempre); en la ConvNeXt no compartida **emerge con la profundidad** (5% → 96% de h).

### 6.2 Dinámica sin el canal (tras corregir el bug: ahora se elimina de **todas** las métricas)

| Métrica | shared | shared −ch236 | ConvNeXt R100 | ConvNeXt R100 −ch195 | Swin R100 | Swin −ch322 |
|---|---|---|---|---|---|---|
| Rectitud R | 0.976 | **0.795** | 0.941 | **0.495** | 0.823 | 0.806 |
| PR_depth | 1.07 | 1.80 | 1.17 | **3.66** | 1.17 | 1.19 |
| PR_spatial | 1.03 | 3.80 | 1.14 | **28.4** | 6.81 | 7.86 |
| κ medio | 0.0005 | 0.0069 | 0.010 | 0.020 | 0.0042 | 0.0044 |
| ω medio | 0.15 | 0.42 | 0.35 | 0.63 | 0.52 | 0.53 |
| a_n/‖a‖ | 0.58 | **0.94** | 0.85 | **0.96** | 0.91 | 0.91 |
| cos(x₀,x_T) | 0.64 | 0.24 | 0.50 | 0.20 | 0.08 | 0.09 |
| cos(h₀,h_T) | 0.75 | −0.05 | 0.17 | 0.01 | 0.04 | 0.03 |

**Shared −ch236 (t de 0 a 9):**
- Dinámica **rotacional**: >90% de la aceleración es normal; `h` termina ortogonal a h₀.
- ω y κ con forma de U: ω 0.74→0.31 (t≈3–4)→0.42→0.26; κ 0.014→0.0035→0.0077.
- La fase "acelerar-frenar" era del canal: sin él a_t/‖a‖ ≈ +0.1/+0.4 → 0 → −0.5 → −0.16.
- Deriva que se apaga: ‖h‖ 53→87→43, ‖x‖ 126→481, ‖h‖/‖x‖ 0.42→0.09. Distancia relativa a la red nativa estable (~0.18).

**ConvNeXt R100 −ch195:**
- ω intra-bloque **constante** (0.55–0.70) en los 9 bloques (con el canal parecía decaer a 0.17: enmascaramiento).
- En cada cambio de bloque, h gira **75–85° en un solo paso** (ω≈130–150) en todos los bloques.
- κ intra 0.043→0.006; a casi toda normal; cos(x,h) en dientes de sierra 0.2→0.6 dentro de cada bloque.
- No se satura (‖h‖/‖x‖≈0.5 constante) y **se aleja de la trayectoria nativa** (dist. rel. 0.23→0.42, creciente).

**Swin**: el canal solo pesa ~17% ⇒ quitarlo apenas cambia nada. **La "rectitud" de Swin es real, la de ConvNeXt no.**

### 6.3 Cómo leer la descomposición tangencial / normal de la aceleración

Ejemplo: `convnext_R100_ES0.01_c289_n1_ignore1`, paneles `metrics/dynamics/acceleration_decomposition_no_interblock.png` y `tangential_acceleration_no_interblock.png`. En ellos ‖a‖ y a_n son casi iguales, mientras que a_t es mayormente negativa y aun así no parece "restar" nada.

**Definición** (`acceleration_decomposition` en `feature_map_explorer.py`), con `a = (h_{d+1} − h_d)/ES` y `v` la velocidad (punto medio):
- a_t = ⟨v, a⟩/‖v‖: proyección de `a` sobre la dirección de `v`. Es un escalar **con signo**.
- a_n = √(‖a‖² − a_t²): longitud de la parte de `a` perpendicular a `v`. Es una norma, así que siempre es ≥ 0.
- Como las dos partes son ortogonales, ‖a‖² = a_t² + a_n² (Pitágoras). a_t entra **al cuadrado**: su signo desaparece y no se resta de a_n.

**Por qué a_n ≈ ‖a‖ aunque a_t no sea pequeña.** Si a_t/‖a‖ = r, entonces a_n/‖a‖ = √(1 − r²). Con r = 0.5 da 0.87 y con r = 0.2 da 0.98. Valores leídos de la gráfica:

| t | ‖a‖ | a_t | a_n |
|---|---|---|---|
| 0 | ≈ 14 | ≈ −7 | ≈ 12.1 |
| 8 | ≈ 124 | ≈ −43 | ≈ 116 |

En t=0 la tangencial es la mitad de la aceleración, y aun así la normal es el 87% de ‖a‖. Las curvas de a_n y ‖a‖ solo se separan claramente cuando |a_t| se acerca a ‖a‖.

**Significado del signo.** a_t es la derivada del módulo de la velocidad: d‖v‖/dt = ⟨v, v̇⟩/‖v‖ = a_t.
- a_t < 0: la trayectoria **frena** (el update residual ‖h‖ se encoge).
- a_t > 0: la trayectoria acelera.
- a_n mide cuánto **gira** la dirección del movimiento; la curvatura es κ = a_n/‖v‖².

En este run, dentro de cada bloque la trayectoria empieza frenando con fuerza y cada vez frena menos (coherente con la contracción de ‖h‖ intra-bloque de §4.1). En los bloques 7 y 8, a_t cruza el cero al final y pasa a acelerar un poco. Al mismo tiempo gira mucho: casi toda la aceleración se gasta en cambiar de dirección y una parte menor en cambiar de velocidad.

**Dimensionalidad.** La componente tangencial vive en **una sola** dirección (la de `v`); la normal agrupa todo el complemento ortogonal (≈ dim − 1, unas 300K dimensiones). Por eso a_n no tiene signo: es la longitud de un vector en ese subespacio, no una dirección concreta. Esto hace que el a_t observado sea notable: si `a` apuntara en una dirección aleatoria en D ≈ 3·10⁵ dimensiones, su proyección sobre una dirección fija sería del orden de ‖a‖/√D ≈ 0.002·‖a‖. Aquí |a_t|/‖a‖ ≈ 0.3–0.5 al inicio de cada bloque, más de 100× lo esperable por azar. El frenado al inicio de cada bloque es un efecto real y muy alineado con la velocidad, aunque en el panel de descomposición parezca que apenas afecta a a_n.

Nota: las curvas son medias sobre muestras/posiciones, y la media de a_n no cumple Pitágoras exactamente con las medias de ‖a‖ y a_t. La diferencia es pequeña y no cambia la interpretación.

**Conclusión**: la apariencia de flujo recto y 1-D de las ConvNeXt es un artefacto de un canal masivo. El resto de la representación evoluciona con una dinámica genuinamente rotacional y multidimensional; el shared la "amortigua" con la profundidad, la no compartida no.

---

## 7. Experimento 6 — ¿Qué papel tiene el canal masivo? (zeroing en shared)

`scripts/zero_outlier_channel_shared.py`: el canal se fuerza a 0 tras **cada** paso residual del stage 3 (dinámica alterada), y ahora también se excluye de las métricas.

### 7.1 Accuracy (ImageNet val, 50k)

| Config | top-1 | loss |
|---|---|---|
| R1 (D9 ES1) baseline | 80.34 | 0.896 |
| R1 zero ch236 | **67.81** (−12.5) | 1.595 |
| R1 zero canal aleatorio (n=20) | ≈80.3 (loss media 0.900, máx 0.914) | |
| R10 (D90 ES0.1) baseline | 80.14 | 0.904 |
| R10 zero ch236 | **67.59** (−12.6) | 1.590 |
| R10 zero canal aleatorio (n=20) | loss media 0.908, máx 0.924 | |

(Son **12.5 pts**, no 14.)

### 7.2 Dinámica: zeroing en la ODE vs ignorar al medir

R10 con ch236 a cero **dentro de la ODE** vs shared D100 dejando evolucionar el canal pero ignorándolo al medir (mismo T=9):

| | R10 zeroing | D100 `_ignore1` |
|---|---|---|
| R | 0.7944 | 0.7947 |
| L | 588.7 | 587.9 |
| PR_depth / PR_spatial | 1.800 / 3.805 | 1.798 / 3.801 |
| κ media | 0.0069 | 0.0069 |

Todas las curvas (‖h‖, ‖x‖, ω, κ, a_t, cos(x,a), cos(h₀,h)) coinciden con <0.5% de diferencia.

### 7.3 Interpretación

Hechos:
1. El canal **no realimenta** la dinámica del resto de canales dentro del stage 3: su presencia o ausencia no cambia la trayectoria de los otros 383 canales.
2. Aun así, quitarlo cuesta 12.5 pts; quitar cualquier otro canal, ~0.
3. Su norma crece **monótonamente** con t (shared: 162→3301), y el campo lo empuja en una dirección casi fija.
4. La accuracy del shared depende fuertemente del horizonte T (§3.2).

Hipótesis:
- **❌ Cronómetro *interno*** (que el campo lea para saber en qué t está): descartada; si el campo lo leyera, borrarlo cambiaría la trayectoria de los demás canales.
- **❓ Cronómetro / escala leída *downstream***: el stage 4 (o el LayerNorm del downsampling) podría usar ‖x₂₃₆‖ como codificación de t o como escala global. Al dominar el 98% de la energía, fija la varianza del LayerNorm de canal en el downsampling; si se quita, la escala del resto cambia ~7× (3301/481) ⇒ shift de distribución en el stage 4. Es el mecanismo típico de las *massive activations* (actúan como bias/escala casi constante, cf. Sun et al. 2024).
- **❓ Bias constante vs información**: no sabemos aún si el valor del canal depende de la imagen (información) o es casi constante (bias).

Tests que discriminan (no ejecutados, ver §10):
- Sustituir el canal por su **media sobre el dataset** (constante) en lugar de 0 → si se recupera la accuracy, es un bias/escala, no información.
- Sustituirlo por su valor a **otro t** (p. ej. t=4.5 o t=18) dejando el resto intacto → si la accuracy cae como en el sweep de T, es un reloj leído downstream.
- **Zero solo a la salida** del stage 3 (no en cada paso): dado el hecho 1, debería dar la misma loss que el zeroing completo; confirma que todo el efecto es downstream.
- Varianza del canal entre imágenes y *linear probe* de la clase sobre él.
- Estadísticas del LayerNorm del downsampling con/sin el canal; reescalar el resto para compensar.

---

## 8. Experimento 7 — Euler vs RK2 vs RK4

### 8.1 Interpolados (plain), top-1 %

| Modelo | RK1 R1 | RK2 R1 | RK4 R1 | RK1 R128 | RK2/RK4 R≥2 |
|---|---|---|---|---|---|
| ConvNeXt | **80.96** | 80.23 | 80.23 | 80.24 | 80.23 |
| ConvNeXt dp0 | **79.27** | 76.73 | 76.98 | 77.02 | 76.96–77.07 |
| Swin-T | **81.08** | 80.29 | 80.33 | 80.34 | 80.31–80.34 |
| ResNet-50 | **80.34** | 78.94 | 79.42 | 79.45 | 79.41–79.44 |
| ResNet-101 | **81.67** | 80.95 | 81.06 | 81.06 | 81.04–81.07 |

### 8.2 Shared (T=9), ver tabla §3.2
- **Pasos grandes** (D≤4): RK4 ≫ Euler (D=4: 78.9 vs 67.8; D=3: 71.4 vs 37.5). Euler en D=2 (ES=4.5) es inestable (5.9%).
- Convergencia al límite 80.12: RK4 desde D≈5, RK2 desde D≈18, Euler desde D≈256.

**Análisis**
- Todas las redes están **afinadas al error de Euler con ES=1**: cualquier solver más preciso en el paso de entrenamiento ya da el límite continuo y pierde 0.15–2.5 pts. Euler ES=1 no es "una mala aproximación de la ODE": es *la* función entrenada; la ODE es una aproximación ligeramente peor.
- Con pasos más grandes que el de entrenamiento, el orden del solver sí importa (RK4 permite D=5 sin pérdida en el shared ⇒ potencial ahorro de cómputo si se integra con menos bloques).

---

## 9. Experimento 8 — Otras interpolaciones (bilinear)

Top-1 % con interpolación **bilinear** de pesos entre bloques consecutivos (Euler):

| R | ConvNeXt | ConvNeXt dp0 | Swin-T | ResNet-50 | ResNet-101 |
|---|---|---|---|---|---|
| 1 | 80.96 | 79.27 | 81.08 | 80.34 | 81.67 |
| 2 | 78.94 | 72.43 | 78.93 | 74.95 | 72.55 |
| 4 | 76.16 | 63.13 | 75.42 | 70.60 | 62.42 |
| 10 | 74.42 | 58.75 | 73.28 | 68.58 | 58.73 |
| 128 | 73.66 | 57.44 | 72.14 | 67.77 | 57.73 |
| **Δ vs plain R128** | **−6.6** | **−19.6** | **−8.2** | **−11.7** | **−23.3** |

RK2/RK4 con bilinear dan lo mismo o peor (convergen al mismo límite bajo).

**Análisis**
- La interpolación lineal en **espacio de pesos** genera bloques intermedios que nunca se entrenaron. Bloques consecutivos no están "alineados" (simetría de permutación de neuronas), así que el punto medio de dos bloques no es un bloque intermedio funcional.
- El daño ordena a los modelos por suavidad entre bloques consecutivos: ConvNeXt (dp 0.1) y Swin son las más suaves; ResNet-101 y ConvNeXt dp0 las menos. Otra pista a favor de que el droppath suaviza el campo en profundidad.
- En las métricas, bilinear dispara ω (~2.0–3.2 vs 0.35–0.67 en plain) y la distancia a la trayectoria nativa (0.31–0.58 vs 0.12–0.24).
- **No hemos probado otras interpolaciones que podrían funcionar**:
  - Interpolar en **espacio de funciones**: `h = (1−α)·f_k(x) + α·f_{k+1}(x)` (no requiere alineamiento de pesos).
  - Interpolación de pesos **tras alinear** bloques (weight matching tipo Git Re-Basin).
  - Para el shared no hace falta: el campo ya es suave por construcción.

---

## 10. ¿Qué falta? Tests pendientes

### 10.1 Ya identificados por ti
- **Shared D27** (33/300 épocas, 70.5% val) y **shared all-stages** (53/300 épocas, 69.95%): terminar el entrenamiento y pasarles la ablation D/ES/RK y las métricas de dinámica.
- **Todo lo de los deltas** (`convnextv1_deltav0_*`: 80.41 / 80.33 (9→27) / 81.25 (wu10 e50 lr1e-3) / 79.16 (wu5 e100 lr1e-4, 30 ep)).

### 10.2 Re-ejecuciones necesarias por el bug del canal ignorado
Hasta el fix, los `_ignore1` solo enmascaraban `norm_mean_*` / `acc_mean_h`. **Están corregidos solo 3**: shared D100 ES0.09, ConvNeXt R100 ES0.01, Swin R100 ES0.01. Quedan **41 runs desactualizados**, entre ellos:
- ResNet-50, ResNet-101 y ConvNeXt dp0 **R100 ES0.01** (estaban en tu tanda pero no se regeneraron; probablemente se interrumpió).
- Todos los `baseline_R1`, `ES0.1`, `ES1`, `bilinear`, `RK4` de todos los modelos, y shared D9 / D100 RK4 / ES0.9 / ES9.

Bastaría con `--force --only <runs _ignore1>` (o `--metrics-only` si no quieres vídeos). Los runs base de ConvNeXt dp0 y los bilinear (29-sep) además no tienen κ ni la descomposición de la aceleración: re-ejecutar con `--metrics-only`.

### 10.3 Experimentos que faltan para cerrar conclusiones
1. **Estadística**: todas las métricas de dinámica son de **1 imagen**. Repetir con n≥32 imágenes de varias clases antes de dar los números como definitivos.
2. **Papel del canal masivo** (§7.3): reemplazo por media, *time-shift*, zero solo a la salida, varianza entre imágenes, linear probe, análisis del LayerNorm del downsampling.
3. **Zeroing en ConvNeXt no compartida, ConvNeXt dp0 y Swin** (solo se ha hecho en shared). En la no compartida el canal emerge con la profundidad: ¿también es "pasivo" respecto a los demás?
4. **Causalidad del droppath**: tasas intermedias, shared con dp=0, y comprobar la receta de los pesos torchvision de ResNet/Swin.
5. **Interpolación en espacio de funciones** y con pesos alineados (§9).
6. **Controles con pesos aleatorios** (`convnext_rand_*`, 8 runs): existen pero no se han analizado en este informe.
7. **Barrido fino del horizonte T** del shared (sección D de `shared_convnext_ablation.py`, 36 configs + 4 de D100): las etiquetas "D=45" ya están corregidas.
8. Los histogramas de Δloss en `summary.json` apuntan a `/leonardo_work/...`; regenerarlos o copiarlos en local si los quieres en el repo.
