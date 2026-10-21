from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"id", "rn"}
LOAN_CATEGORY_COLUMNS = (
    "enc_loans_account_holder_type",
    "enc_loans_credit_status",
    "enc_loans_credit_type",
    "enc_loans_account_cur",
)


def _flatten(columns: pd.MultiIndex) -> list[str]:
    return [f"{column}__{stat}" for column, stat in columns]


def _binary_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
        and (column.startswith("is_zero_") or column.endswith("_flag"))
    ]


def _payment_columns(df: pd.DataFrame) -> list[str]:
    def payment_index(column: str) -> int:
        return int(column.rsplit("_", 1)[1])

    columns = [column for column in df.columns if column.startswith("enc_paym_")]
    return sorted(columns, key=payment_index)


def _cast_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan).reset_index()
    feature_cols = [column for column in frame.columns if column != "id"]
    frame[feature_cols] = frame[feature_cols].astype("float32")
    return frame


def aggregate_legacy_credit_history(
    history: pd.DataFrame,
    *,
    include_latest: bool = True,
) -> pd.DataFrame:
    """Reproduce the original M1 aggregation for a fair CV ablation.

    This function intentionally preserves M1's mechanically generated numeric
    statistics. It is not presented as the preferred representation; its role
    is to quantify how much predictive signal those legacy features contain.

    With the Alfa-Bank training schema (59 non-ID columns), this produces
    414 features when ``include_latest=True`` and 355 without ``latest``.
    """
    if "id" not in history.columns:
        raise ValueError("Expected column 'id'.")

    df = history.copy()
    numeric_cols = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in IDENTIFIER_COLUMNS
    ]
    if not numeric_cols:
        return df[["id"]].drop_duplicates().reset_index(drop=True)

    grouped = df.groupby("id", sort=False, observed=True)
    result = grouped.size().rename("history__row_count").to_frame()

    stats = grouped[numeric_cols].agg(["mean", "std", "min", "max", "sum", "nunique"])
    stats.columns = _flatten(stats.columns)
    result = result.join(stats)

    if include_latest:
        if "rn" not in df.columns:
            raise ValueError("M1 latest features require column 'rn'.")
        latest = (
            df.sort_values(["id", "rn"], kind="stable")
            .groupby("id", sort=False, observed=True)[numeric_cols]
            .last()
        )
        latest.columns = [f"{column}__latest" for column in latest.columns]
        result = result.join(latest)

    return _cast_feature_frame(result)


def aggregate_credit_history(history: pd.DataFrame) -> pd.DataFrame:
    """M2 typed application-level features.

    Design choices:
    - no ``latest by rn`` feature: rn is treated only as a row/order identifier;
    - binary indicators -> counts/rates;
    - ``enc_*`` values -> categorical diversity (nunique), avoiding arithmetic
      over arbitrary category codes;
    - ``pre_*`` binned codes -> compact distribution statistics. Their numeric
      ordering is an explicit modelling hypothesis that can later be ablated;
    - basic payment-history dynamics use only equality/change information
      between encoded states, so they do not assume numeric distances between
      category codes.
    """
    if "id" not in history.columns:
        raise ValueError("Expected column 'id'.")

    df = history.copy()
    grouped = df.groupby("id", sort=False, observed=True)
    result = grouped.size().rename("history__row_count").to_frame()

    binary_cols = _binary_columns(df)
    if binary_cols:
        binary_stats = grouped[binary_cols].agg(["mean", "sum", "max"])
        binary_stats.columns = _flatten(binary_stats.columns)
        result = result.join(binary_stats)

    pre_cols = [
        column
        for column in df.columns
        if column.startswith("pre_") and column not in binary_cols
    ]
    if pre_cols:
        pre_stats = grouped[pre_cols].agg(["mean", "std", "min", "max", "nunique"])
        pre_stats.columns = _flatten(pre_stats.columns)
        result = result.join(pre_stats)

    encoded_cols = [
        column
        for column in df.columns
        if column.startswith("enc_") and column not in binary_cols
    ]
    if encoded_cols:
        encoded_stats = grouped[encoded_cols].nunique(dropna=True)
        encoded_stats.columns = [f"{column}__nunique" for column in encoded_stats]
        result = result.join(encoded_stats)

    payment_cols = _payment_columns(df)
    if len(payment_cols) >= 2:
        payment_values = df[payment_cols]
        df["_payment_state_nunique"] = payment_values.nunique(axis=1, dropna=True)
        df["_payment_transition_count"] = (
            payment_values.iloc[:, 1:].to_numpy()
            != payment_values.iloc[:, :-1].to_numpy()
        ).sum(axis=1)

        payment_grouped = df.groupby("id", sort=False, observed=True)[
            ["_payment_state_nunique", "_payment_transition_count"]
        ]
        payment_stats = payment_grouped.agg(["mean", "std", "max"])
        payment_stats.columns = [
            f"payment{column.replace('_payment', '')}__{stat}"
            for column, stat in payment_stats.columns
        ]
        result = result.join(payment_stats)

    loan_category_cols = [column for column in LOAN_CATEGORY_COLUMNS if column in df.columns]
    if loan_category_cols:
        unique_profiles = (
            df[["id", *loan_category_cols]]
            .drop_duplicates()
            .groupby("id", sort=False, observed=True)
            .size()
            .rename("loan_categories__unique_profiles")
        )
        result = result.join(unique_profiles)

    return _cast_feature_frame(result)


def aggregate_category_distributions(
    history: pd.DataFrame,
    columns: Iterable[str] = LOAN_CATEGORY_COLUMNS,
) -> pd.DataFrame:
    """Application-level category fractions without treating codes as magnitudes.

    For each selected encoded column and each category observed in the raw
    chunk, create a feature such as ``enc_loans_credit_status__frac__3``.
    Fractions for one source column sum to one for each application (ignoring
    missing source values).
    """
    if "id" not in history.columns:
        raise ValueError("Expected column 'id'.")

    selected = [column for column in columns if column in history.columns]
    base_ids = history[["id"]].drop_duplicates().set_index("id")
    result = base_ids.copy()

    for column in selected:
        non_missing = history[["id", column]].dropna()
        if non_missing.empty:
            continue

        counts = non_missing.groupby(["id", column], sort=False, observed=True).size()
        fractions = counts.div(counts.groupby(level=0).sum())
        wide = fractions.unstack(fill_value=0.0)
        wide = wide.reindex(base_ids.index, fill_value=0.0)
        wide = wide.reindex(columns=sorted(wide.columns, key=lambda value: str(value)))
        wide.columns = [f"{column}__frac__{_format_category(value)}" for value in wide.columns]
        result = result.join(wide, how="left")

    result = result.fillna(0.0)
    return _cast_feature_frame(result)


def _format_category(value: object) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).replace(" ", "_")


def aggregate_extended_payment_dynamics(history: pd.DataFrame) -> pd.DataFrame:
    """Direction-agnostic payment-sequence features for M2.1.

    The features use equality, transitions, run lengths and categorical
    composition. They intentionally avoid subtraction/averaging of encoded
    payment-state values and therefore do not assume metric distances between
    category codes. Sequence reversal leaves the core dynamics unchanged.
    """
    if "id" not in history.columns:
        raise ValueError("Expected column 'id'.")

    payment_cols = _payment_columns(history)
    base = history[["id"]].drop_duplicates().set_index("id")
    if len(payment_cols) < 2:
        return _cast_feature_frame(base)

    values = history[payment_cols].to_numpy()
    n_rows, n_steps = values.shape

    # Dataset columns are integer encoded. The code below still handles NaN if
    # future data versions introduce missing payment states.
    valid = ~pd.isna(values)
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    transitions = (values[:, 1:] != values[:, :-1]) & adjacent_valid
    comparable_pairs = adjacent_valid.sum(axis=1)
    transition_count = transitions.sum(axis=1).astype("float32")
    transition_rate = np.divide(
        transition_count,
        comparable_pairs,
        out=np.zeros(n_rows, dtype="float32"),
        where=comparable_pairs > 0,
    )

    state_nunique = pd.DataFrame(values).nunique(axis=1, dropna=True).to_numpy(dtype="float32")

    # Longest constant-state run. Missing values break a run.
    current_run = np.zeros(n_rows, dtype=np.int16)
    longest_run = np.zeros(n_rows, dtype=np.int16)
    previous = None
    previous_valid = None
    for step in range(n_steps):
        current = values[:, step]
        current_valid = valid[:, step]
        if step == 0:
            current_run = current_valid.astype(np.int16)
        else:
            same = current_valid & previous_valid & (current == previous)
            current_run = np.where(current_valid, np.where(same, current_run + 1, 1), 0)
        longest_run = np.maximum(longest_run, current_run)
        previous = current
        previous_valid = current_valid

    endpoint_same = (
        valid[:, 0] & valid[:, -1] & (values[:, 0] == values[:, -1])
    ).astype("float32")

    # Modal-state concentration per loan/credit history row. We scan only the
    # globally observed states and keep one length-n_rows accumulator.
    observed_states = pd.unique(history[payment_cols].to_numpy().ravel())
    observed_states = sorted(
        (state for state in observed_states if not pd.isna(state)),
        key=lambda state: str(state),
    )
    max_state_count = np.zeros(n_rows, dtype=np.int16)
    for state in observed_states:
        count = (values == state).sum(axis=1)
        max_state_count = np.maximum(max_state_count, count.astype(np.int16))
    valid_count = valid.sum(axis=1)
    mode_fraction = np.divide(
        max_state_count,
        valid_count,
        out=np.zeros(n_rows, dtype="float32"),
        where=valid_count > 0,
    )

    row_features = pd.DataFrame(
        {
            "id": history["id"].to_numpy(),
            "_m21_payment_state_nunique": state_nunique,
            "_m21_payment_transition_count": transition_count,
            "_m21_payment_transition_rate": transition_rate,
            "_m21_payment_longest_constant_run": longest_run.astype("float32"),
            "_m21_payment_mode_fraction": mode_fraction,
            "_m21_payment_endpoint_same": endpoint_same,
        }
    )

    grouped = row_features.groupby("id", sort=False, observed=True)
    stats = grouped[
        [
            "_m21_payment_transition_rate",
            "_m21_payment_longest_constant_run",
            "_m21_payment_mode_fraction",
            "_m21_payment_endpoint_same",
        ]
    ].agg(["mean", "std", "min", "max"])
    stats.columns = [
        f"payment_ext{column.replace('_m21_payment', '')}__{stat}"
        for column, stat in stats.columns
    ]

    # Overall application-level composition of encoded payment states across
    # all credits and all payment positions. This is a category histogram, not
    # an arithmetic operation on the state codes.
    denominator = (
        pd.Series(valid.sum(axis=1), index=history["id"].to_numpy())
        .groupby(level=0, sort=False)
        .sum()
        .astype("float64")
    )
    for state in observed_states:
        matches_per_row = (values == state).sum(axis=1).astype("int16")
        state_counts = (
            pd.Series(matches_per_row, index=history["id"].to_numpy())
            .groupby(level=0, sort=False)
            .sum()
        )
        fractions = state_counts.div(denominator).rename(
            f"payment_overall_state__frac__{_format_category(state)}"
        )
        stats = stats.join(fractions, how="outer")

    stats = stats.reindex(base.index)
    return _cast_feature_frame(stats)


def aggregate_canonical_m21_features(history: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen M2.1-E representation used from M3 onward.

    This is a convenience composition of the typed M2 block, loan-category
    distributions and direction-agnostic extended payment dynamics.
    """
    typed = aggregate_credit_history(history)
    catdist = aggregate_category_distributions(history)
    payment = aggregate_extended_payment_dynamics(history)
    return join_feature_blocks(typed, catdist, payment)



def join_feature_blocks(*blocks: pd.DataFrame) -> pd.DataFrame:
    """Merge application-level feature blocks and reject accidental duplicates."""
    if not blocks:
        raise ValueError("At least one feature block is required.")

    result = blocks[0]
    for block in blocks[1:]:
        overlapping = (set(result.columns) & set(block.columns)) - {"id"}
        if overlapping:
            raise ValueError(f"Duplicate feature names across blocks: {sorted(overlapping)[:5]}")
        result = result.merge(block, on="id", how="outer", validate="one_to_one")
    return result
