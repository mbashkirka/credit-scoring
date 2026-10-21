# Architecture

## 1. Experiment and governance flow

```mermaid
flowchart TD
    R[Raw Kaggle credit history] --> M1[M1 baseline]
    M1 --> M21[M2/M2.1 typed features + ablation]
    M21 --> M3[M3 CatBoost vs Logistic]
    M3 --> M31[M3.1 controlled tuning]
    M31 --> M32[M3.2 frozen external validation\nchunks 1-3]
    M32 --> F[Freeze base configuration]
    F --> T[M4 train CatBoost\nchunks 0-7]
    T --> C8[M4 calibration fit\nchunk 8]
    C8 --> C9[M4 calibration selection\nchunk 9]
    C9 --> P[M5 freeze bands + balanced policy\nchunks 8+9]
    P --> L[M6 one-shot lockbox\nchunks 10+11]
    L --> X[M7 explanations]
    L --> API[M8 serving]
    API --> D[M9 Docker + CI]
    L --> MON[M10 historical monitoring replay]
    MON --> REL[M11 portfolio release]
```

### Governance invariants

- M3.2 evaluates frozen M3/M3.1 candidates on chunks 1–3 without refitting.
- M4 chooses the calibration method on chunk 9 after fitting candidate calibrators on chunk 8.
- M5 policy/bands are frozen before M6.
- M6 opens chunks 10–11 once; afterward they are historical evaluation data.
- M10 thresholds were introduced after M6, so the M6 replay is retrospective, not a prospective alerting claim.

## 2. Offline feature/model path

```mermaid
flowchart LR
    A[Account-level history rows] --> B[Typed feature aggregation]
    B --> C[Schema alignment\n212 frozen features]
    C --> D[FrozenMedianImputer]
    D --> E[CatBoost Model V1]
    E --> F[raw score / RawFormulaVal]
    E --> G[raw class-weighted probability]
    G --> H[Isotonic calibrator]
    H --> I[calibrated PD]
    I --> J[Risk band A-E]
    I --> K[Balanced decision policy]
```

`rn` is not assumed to be a real timestamp. The canonical M2.1 representation deliberately avoids the earlier `latest-by-rn` block.

## 3. Online serving path

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Features as Shared feature code
    participant Model as CatBoost + calibrator
    participant Policy as Band/decision layer
    participant XAI as TreeSHAP

    Client->>API: POST /v1/score raw credit_history
    API->>API: Pydantic schema validation
    API->>Features: aggregate_canonical_m21_features
    Features-->>API: 212 aligned features
    API->>Model: impute + CatBoost + isotonic
    Model-->>API: raw score + calibrated PD
    API->>Policy: PD
    Policy-->>API: credit score + band + decision
    API->>XAI: optional native TreeSHAP
    XAI-->>API: semantic reason groups
    API-->>Client: versioned response + artifact fingerprint
```

The API never accepts the target label. Batch explanations are disabled by default because TreeSHAP adds latency.

## 4. Monitoring architecture

```mermaid
flowchart LR
    REF[Reference chunks 8+9] --> RP[Fixed feature/PD/schema profiles]
    CUR[Current scoring window] --> S[Schema + missingness]
    CUR --> F[Feature PSI]
    CUR --> P[PD / band / decision drift]
    RP --> S
    RP --> F
    RP --> P
    Y[Delayed matured outcomes] --> O[AUC / AP / KS / Brier / LogLoss\ncalibration bias & slope]
    B[Chunk 9 pre-deployment benchmark] --> O
    S --> A[Alert policy]
    F --> A
    P --> A
    O --> A
    A --> Z[PASS / WARNING / CRITICAL]
```

Monitoring explicitly separates **label-free signals** from metrics that require matured outcomes. This matters in the historical M10 replay: PD distribution and decision mix were stable while discrimination declined.

## 5. Deployment boundary

The production Docker image contains code plus only seven frozen serving artifacts. It excludes raw data, training caches, research plots, and other experiment artifacts. The application runs as UID/GID 10001, and readiness depends on successful artifact/runtime loading.
