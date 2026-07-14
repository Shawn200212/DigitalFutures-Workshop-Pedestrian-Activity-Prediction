# Presentation Script

**From Partial Street-View Labels to Network-Wide Pedestrian Activity Prediction**
*A GraphSAGE-Based Framework for Urban Renewal Diagnosis*

Team **The Capture** — DigitalFUTURES Workshop · total runtime ≈ 12:40

Slide images: [`slides/`](slides/) (23 slides, exported from the final deck).

---

## Slide 1 (0:00–0:45) — Title

Good morning everyone. Our project is titled *From Partial Street-View Labels to Network-Wide Pedestrian Activity Inference*.

What we are trying to understand is quite simple to state, but difficult to solve in practice: if we only observe part of a city through street-view images, can we still infer how pedestrian activity is distributed across the entire urban network?

In other words, **can partial visibility lead to full-network understanding?**

## Slide 2 (0:45–1:10) — Team

*(Slide image withheld from this repository to protect team members' privacy.)*

This is our team of seven members. Our work is driven by a shared question: what do we know about streets that are not directly visible in data?

## Slide 3 (1:10–2:05) — Uneven urban data

Urban data today is highly uneven. We tend to observe major roads, commercial corridors, and central districts, because those areas are well-covered by sensors or imagery. In contrast, back streets, residential lanes, and older neighborhoods are often missing from observation.

The important point is that these invisible areas are not necessarily inactive. They are simply **under-observed**, which creates a bias in how we understand cities.

## Slide 4 (2:05–3:00) — Limitations of existing data

Field surveys are accurate but cannot scale to a whole city. Mobile signal data scales, but is often inaccessible due to privacy and policy restrictions. Cameras observe continuously, but only in limited spatial coverage. Street-view imagery is the most scalable option we have — yet even it does not cover every street consistently.

So there is still a missing piece: **a full-network view of pedestrian activity.**

## Slide 5 (3:00–3:30) — Research question

Can partial street-view observations be used to infer pedestrian activity across unobserved street networks? More specifically, can we learn a mapping from observed streets to the unseen parts of the city?

## Slide 6 (3:30–4:15) — Study area & tiling

We study Shanghai within the Outer Ring Road, divided into **100 spatial graph tiles**. Each tile contains a core region, where predictions are evaluated, and a buffer region, which preserves spatial context during message passing in the GNN. This design avoids edge effects in spatial learning.

## Slide 7 (4:15–5:00) — Graph representation

Each road segment is a node; intersections define edges. Node features come from the built environment, while the target label is derived from pedestrian activity estimated from street-view images — connecting physical urban structure with observed human activity.

## Slide 8 (5:00–5:50) — Features & model

Four feature categories: **road geometry** (length, orientation, curvature); **accessibility** (metro distance, bus stop density); **built environment** (building density, height, land-use mix); **network topology** (betweenness, centrality). We use GraphSAGE so the model aggregates information from neighboring streets instead of treating each road independently.

## Slide 9 (5:50–6:10) — Dataset

100 graph tiles, split into training / validation / test **at the graph level**, ensuring spatial generalization.

## Slide 10 (6:10–6:55) — Labels, not inputs

A key design choice: street-view images are **only used for generating labels, not as model inputs**. Each panorama is processed from multiple directions, pedestrians are detected, and results are aggregated at the road-segment level into a relative measure of pedestrian activity.

## Slide 11 (6:55–7:25) — Training

The model begins to overfit after several epochs; we apply early stopping and select epoch 10 as the best validation point.

## Slide 12 (7:25–8:00) — Regression results

On the test set the model achieves a **Pearson correlation of 0.554** — it captures the overall spatial structure of pedestrian activity, though it underestimates high-activity streets, suggesting extreme urban conditions are not fully captured by the current features.

## Slide 13 (8:00–8:45) — External validation with POI

Areas predicted as high-activity strongly align with commercial facilities, transport nodes, and public services — the model is not fitting noise, but learning a meaningful relation between urban structure and human activity.

## Slide 14 (8:45–9:30) — Error analysis

Errors are not random: they concentrate in high-intensity areas, especially around transport hubs and large commercial complexes — places driven by functional and temporal dynamics that our feature system does not fully represent.

## Slide 15 (9:30–10:00) — Task reformulation

Instead of predicting exact values, we shift to a binary task: identifying **high-vitality vs low-vitality** streets — better aligned with practical planning decisions.

## Slide 16 (10:00–10:30) — Threshold tuning

A threshold of **1.12** balances multiple metrics best; recall for high-vitality streets reaches **74.5%**, suitable for screening applications.

## Slide 17 (10:30–11:15) — Spatial case analysis

Best cases: active corridors identified accurately. Typical cases: overall spatial pattern correct with local errors. Challenging cases: extreme hotspots are hard to predict — failures consistent with missing functional information.

## Slide 18 (11:15–11:40) — Planning diagnosis framework

Combining predicted and observed values classifies streets into categories — including **renewal-priority areas** where predictions are high but observations are low.

## Slide 19 (11:40–12:05) — Planning applications

Identify high-vitality clusters, detect underperforming but promising streets, and prioritize field investigation.

## Slides 20–21 (12:05–12:30) — Contribution

The framework translates partial street-level observations into network-wide urban understanding — revealing how urban structure shapes pedestrian activity at scale, while highlighting where structural explanations fall short.

## Slide 22 (12:30–12:40) — Closing

**The streets that were once invisible are now becoming interpretable within a unified analytical framework.**
