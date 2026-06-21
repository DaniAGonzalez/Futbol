# Futbol
# Football Analytics Portfolio — Daniela González

Portafolio de dos proyectos de *football data & insights* construidos de punta a punta: desde la ingesta de datos crudos hasta una capa de decisión (shortlists, valuaciones, one-pagers). El foco es mostrar el pipeline completo de modelado —clustering, modelos bayesianos, embeddings de grafo, valor de mercado— aplicado a problemas reales de scouting y reclutamiento, con criterio metodológico y comunicación honesta de la incertidumbre.

Soy research scientist en machine learning (PhD en Ciencias Biológicas, Máster en Data Science), con experiencia en graph neural networks y modelado probabilístico. Estos proyectos combinan ese background técnico con el análisis de fútbol. LinkedIn / contacto: **@biodatadani**.

Este repositorio contiene los **reportes y visualizaciones** de cada proyecto. El código completo no está publicado; los reportes documentan la metodología, los hallazgos y las decisiones de modelado.

## Los dos proyectos

| | Proyecto | Datos | Foco |
|---|---|---|---|
| 1 | Scouting Intelligence — World Cup 2022 | StatsBomb Open Data (FIFA World Cup 2022) | Pipeline completo de scouting: roles, finishing bayesiano, grafos de pases, shortlist |
| 2 | RiverValue | FBref + Transfermarkt (Liga Profesional Argentina) | Conectar rendimiento con valor de mercado para decisiones de fichajes |

## Proyecto 1 — Scouting Intelligence (World Cup 2022)

Sistema de *decision-support* para scouting construido sobre StatsBomb Open Data. Funciona como el primer entregable de un departamento de datos: un pipeline reproducible que va de los eventos crudos a una shortlist y a one-pagers por jugador, pasando por embeddings, clustering de roles y una capa bayesiana.

El recorrido es: ingesta de eventos y construcción de una *feature matrix* per-90 para 233 jugadores del Mundial; aprendizaje de un espacio latente con PCA como baseline y un autoencoder en PyTorch; descubrimiento de roles por datos con un Gaussian Mixture Model y visualización con UMAP; un motor de similitud por coseno sobre el latente; un modelo bayesiano jerárquico de finishing; embeddings sobre el grafo de pases; y una capa de decisión con shortlist multi-criterio y fichas exportables.

Hallazgos y momentos centrales:

- **El autoencoder valida la señal.** El espacio latente no-lineal coincide con el de PCA, lo que da confianza en que la representación captura estilo de juego real y no artefactos del modelo.
- **El resultado nulo bayesiano es el hallazgo más fuerte.** Modelando el finishing como un Poisson jerárquico con *partial pooling* (reparametrización no-centrada para resolver el funnel), los intervalos de credibilidad de los 22 finalizadores cruzan el 1.0 sin excepción. Con datos de un solo torneo no se puede distinguir habilidad de finalización del azar: un club que paga sobreprecio por el "killer instinct" de un goleador de Mundial está comprando varianza, no habilidad.
- **Diagnóstico honesto de la GNN.** Implementé un Graph Autoencoder con encoder GCN desde cero en PyTorch puro. Al evaluarlo, diagnostiqué en secuencia por qué no aprendía —nodos sin features, luego negative sampling— hasta confirmar la causa raíz: en un grafo de 32 componentes casi completamente conectadas, no hay estructura aprendible por link prediction. node2vec resultó más apropiado para esa topología, y lo documenté en vez de esconderlo.
- **Demo de scouting: Gvardiol.** Para el perfil de Otamendi, la shortlist multi-criterio devuelve #1 a Gvardiol —con 20 años y fichado por ~90M meses después— usando solo datos de eventos, sin edad ni valor de mercado.

Stack: `statsbombpy`, `pandas`/`numpy`, `scikit-learn` (PCA, GMM, StandardScaler), `torch` (autoencoder y GCN-GAE), `pymc` + `arviz`, `node2vec` + `networkx`, `umap-learn`, `mplsoccer` / `matplotlib`. App de exploración en `streamlit`.

Reporte: `reports/scouting_intelligence_wc2022.pdf` · Visualizaciones: `figures/`

## Proyecto 2 — RiverValue (River Plate, Liga Profesional Argentina)

> Sección preliminar, a afinar con el notebook de RiverValue.

Proyecto enfocado en River Plate que conecta el rendimiento de los jugadores con su valor de mercado, para informar decisiones de scouting y de negocio: identificar jugadores subvalorados, evaluar fichajes contra lo pagado y orientar perfiles de reclutamiento.

El pipeline ingesta datos de FBref vía `soccerdata` (con un `league_dict` propio para habilitar la Liga Profesional Argentina), aplica shrinkage bayesiano —Beta-Binomial para finishing y Gamma-Poisson per-90 para evitar inflar a jugadores con pocos minutos— y deriva los pesos de las features con PCA ajustado sobre toda la liga. El valor de mercado de Transfermarkt se integra por CSV y se cruza con la producción.

Hallazgos:

- FBref no provee xG para la Liga Profesional Argentina, lo que condiciona el set de métricas disponibles.
- La producción por sí sola casi no explica el valor de mercado (R²=0.03); al sumar la edad sube a 0.48. El valor está más determinado por el perfil y la edad del jugador que por su rendimiento actual.
- Subvalorados en ataque: Quintero, Freitas, Subiabre.
- Riesgo de fichaje más claro: Páez (fee alto, valor actual por debajo de lo pagado, producción modesta).

Stack: `soccerdata` / FBref, `pymc` + `arviz`, `scikit-learn` (PCA, StandardScaler), `mplsoccer` / `matplotlib`, `pandas`.

Reporte: `reports/rivervalue.pdf` · Visualizaciones: `figures/`

## Notas de metodología

Los dos proyectos comparten una misma postura: comunicar la incertidumbre en vez de esconderla. Las tasas crudas (goles/xG, métricas per-90 con pocos minutos) engañan en muestras chicas, y por eso el shrinkage bayesiano es central en ambos. Cuando un modelo no aprende o un resultado es nulo, eso también es un hallazgo y se documenta como tal.
