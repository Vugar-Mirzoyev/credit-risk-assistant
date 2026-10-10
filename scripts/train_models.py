"""Fit the models and write the validation report.

    python scripts/train_models.py

Six models, because each pair answers a question a single headline number
cannot:

============================  =============================================
champion, no grade            the deployable scorecard
champion, with grade          what LendingClub's own risk opinion adds
challenger, raw               the ceiling a black box reaches
challenger, WOE               the same comparison with inputs held constant,
                              which isolates model class from binning
leaky champion                D2's deliverable: the same scorecard given the
leaky challenger              post-origination columns, so the gap between
                              leaky and clean is measured rather than asserted
============================  =============================================

The leaky pair is fitted deliberately and reported, never deployed. Its point
is that it looks excellent. A reviewer who has only seen the clean numbers has
no way to judge whether they are good; seeing both is what makes the audit
worth anything.

The champion is then exported to `models/champion.json` for the scoring
service (D12), and the whole holdout is scored again through the service's
own code path. The export is refused unless that reproduces the fitted
scorecard: a file the service would score differently from the validated
model is not the validated model. Finally the run is recorded and the
champion registered (D15), unless tracking is switched off in the
configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation.explain import compare_with_scorecard, explain_global
from src.evaluation.metrics import calibration, discrimination, gains_table
from src.evaluation.stability import characteristic_stability, stability_by_quarter
from src.features.woe import WOE_PREFIX
from src.models.challenger import fit_challenger
from src.models.scorecard import GRADE_FAMILY, fit_scorecard, without_grade_family
from src.pipeline import Dataset, prepare, prepare_leaky
from src.serving.artefact import champion_from_dict, export_champion, write_champion
from src.serving.score import score_frame

CHAMPION = "champion, no grade"

#: Largest difference in predicted PD allowed between the fitted scorecard and
#: the exported file scored through `src/serving`. Floating-point noise only:
#: this is a check that the two are the same model, not a modelling choice.
PARITY_TOLERANCE = 1e-9


def _row(name: str, kind: str, features: int, y_train, p_train, y_test, p_test):
    train = discrimination(y_train, p_train)
    test = discrimination(y_test, p_test)
    fit = calibration(y_test, p_test)
    return {
        "model": name,
        "kind": kind,
        "features": features,
        "train_gini": round(train.gini, 4),
        "test_gini": round(test.gini, 4),
        "test_auc": round(test.auc, 4),
        "test_ks": round(test.ks, 4),
        "brier": round(fit.brier, 4),
        "level_error": round(fit.calibration_in_the_large, 4),
        "calib_slope": round(fit.slope, 4),
    }


def evaluate_all(data: Dataset, config: Config) -> tuple[pd.DataFrame, dict]:
    y_train, y_test = data.y_train, data.y_test
    rows: list[dict] = []
    artefacts: dict[str, object] = {}

    clean = list(data.binner.selected_columns)
    for name, features in (
        ("champion, with grade", clean),
        ("champion, no grade", list(without_grade_family(clean))),
    ):
        card, report = fit_scorecard(
            data.woe_train, y_train, data.binner, config.model,
            features=features, name=name,
        )
        rows.append(
            _row(name, "scorecard", len(card.features), y_train,
                 card.predict_proba(data.woe_train), y_test,
                 card.predict_proba(data.woe_test))
        )
        artefacts[name] = (card, report)

    raw = [c for c in data.features if c not in GRADE_FAMILY]
    boosted, boosted_report = fit_challenger(
        data.train, y_train, features=raw, date_column=data.date_column,
        config=config.challenger, model_config=config.model,
        name="challenger, raw",
    )
    rows.append(
        _row("challenger, raw", "boosted", len(raw), y_train,
             boosted.predict_proba(data.train), y_test,
             boosted.predict_proba(data.test))
    )
    artefacts["challenger, raw"] = (boosted, boosted_report)

    champion_columns = artefacts["champion, no grade"][0].features
    woe_columns = [WOE_PREFIX + f for f in champion_columns]
    woe_train_dated = data.woe_train.assign(
        **{data.date_column: data.train[data.date_column].to_numpy()}
    )
    boosted_woe, _ = fit_challenger(
        woe_train_dated, y_train, features=woe_columns,
        date_column=data.date_column, config=config.challenger,
        model_config=config.model, name="challenger, woe",
    )
    rows.append(
        _row("challenger, woe", "boosted", len(woe_columns), y_train,
             boosted_woe.predict_proba(data.woe_train), y_test,
             boosted_woe.predict_proba(data.woe_test))
    )

    champion = artefacts["champion, no grade"][0]
    champion_train = champion.predict_proba(data.woe_train)
    champion_test = champion.predict_proba(data.woe_test)
    artefacts["gains"] = gains_table(y_test, champion_test)
    artefacts["calibration"] = calibration(y_test, champion_test)

    # Stability: the score first, then the inputs. A stable score can hide two
    # features moving in opposite directions, so both are reported.
    artefacts["psi"] = stability_by_quarter(
        champion_train, champion_test, data.test[data.date_column]
    )
    artefacts["csi"] = characteristic_stability(
        data.train, data.test, list(data.features)
    )

    explanation = explain_global(
        boosted, data.test, config=config.explain,
        random_state=config.model.random_state,
    )
    artefacts["shap"] = explanation
    artefacts["shap_vs_points"] = compare_with_scorecard(
        explanation, champion.points_table()
    )
    return pd.DataFrame(rows), artefacts


def evaluate_leaky(config: Config) -> pd.DataFrame:
    """D2's deliverable: the same models, handed the outcome columns.

    Loaded separately from the clean path, because the clean path never reads
    a post-origination column at all. That is the point of it.
    """
    data = prepare_leaky(config)
    features = [c for c in data.features if c not in GRADE_FAMILY]

    card, _ = fit_scorecard(
        data.woe_train, data.y_train, data.binner, config.model,
        name="leaky scorecard",
    )
    rows = [
        _row("leaky scorecard", "scorecard", len(card.features), data.y_train,
             card.predict_proba(data.woe_train), data.y_test,
             card.predict_proba(data.woe_test))
    ]

    boosted, _ = fit_challenger(
        data.train, data.y_train, features=features,
        date_column=data.date_column, config=config.challenger,
        model_config=config.model, name="leaky challenger",
    )
    rows.append(
        _row("leaky challenger", "boosted", len(features), data.y_train,
             boosted.predict_proba(data.train), data.y_test,
             boosted.predict_proba(data.test))
    )
    return pd.DataFrame(rows)


def write_report(
    data_summary: str,
    clean: pd.DataFrame,
    leaky: pd.DataFrame,
    artefacts: dict,
    path: Path,
    config_top: int = 20,
) -> None:
    card, card_report = artefacts["champion, no grade"]
    calib = artefacts["calibration"]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model validation",
        "",
        "Generated by `python scripts/train_models.py`. Every figure here is",
        "measured on the out-of-time holdout unless the column says otherwise.",
        "",
        "## Data path",
        "",
        "```",
        data_summary,
        "```",
        "",
        "## Discrimination and calibration (D4, D5)",
        "",
        clean.to_markdown(index=False),
        "",
        "## Leaky against clean (D2)",
        "",
        "The models below are fitted on the post-origination columns the audit",
        "excludes. They are never deployed and exist only to be compared: the",
        "gap between them and the clean models is the size of the mistake the",
        "whitelist prevents.",
        "",
        leaky.to_markdown(index=False),
        "",
        "## Champion calibration by band",
        "",
        calib.to_markdown(),
        "",
        "## Champion rank ordering",
        "",
        artefacts["gains"].round(4).to_markdown(index=False),
        "",
        "## Stability (D5)",
        "",
        "The champion's predicted PD, quarter by quarter, against its training",
        "distribution. Buckets are fixed on the training window: recomputing",
        "them per quarter would compare every window with itself and never",
        "report a shift.",
        "",
        artefacts["psi"].to_markdown(index=False),
        "",
        "Characteristic stability, the ten features that moved furthest. A",
        "stable score can hide two features drifting in opposite directions,",
        "so the inputs are measured as well as the output.",
        "",
        "`floor_driven` marks an index whose value comes mostly from buckets one",
        "window never reached. The band is still worth reading; the digits are",
        "not, because they depend on the floor rather than on the data. On this",
        "extract every flagged feature belongs to the bureau family LendingClub",
        "only began reporting in 2016, so the shift is a reporting change and",
        "not a change in borrowers - see the known limitations in docs/DESIGN.md.",
        "",
        artefacts["csi"].head(10).to_markdown(index=False),
        "",
        "## Explaining the challenger (SHAP)",
        "",
        "The champion needs none of this: its points table already says what",
        "each attribute contributed. The challenger has no coefficients, so its",
        "output is attributed to the features additively instead.",
        "",
        "```",
        artefacts["shap"].summary(),
        "```",
        "",
        artefacts["shap"].table(config_top).to_markdown(index=False),
        "",
        "Where the two models disagree about what matters:",
        "",
        artefacts["shap_vs_points"].head(15).to_markdown(index=False),
        "",
        "## Champion scorecard",
        "",
        "```",
        card_report.summary(),
        "```",
        "",
        card.points_table().to_markdown(index=False),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_checked(
    data: Dataset,
    clean: pd.DataFrame,
    artefacts: dict,
    *,
    cutoff: object,
    code: dict[str, str],
) -> tuple[dict, float]:
    """D12: the champion as a file, proven to score as the fitted model does.

    The holdout goes through `score_frame` from raw columns - derivation, bin
    lookup and points included - and is compared with the scorecard's own
    predictions on the WOE frame. Anything beyond floating-point noise means
    the service would not be scoring the model this report validates.
    """
    card = artefacts[CHAMPION][0]
    row = clean.set_index("model").loc[CHAMPION]
    issued = data.train[data.date_column]
    metadata = {
        "trained_on": {
            "split_cutoff": str(cutoff),
            "train_rows": int(len(data.train)),
            "train_from": str(issued.min().date()),
            "train_to": str(issued.max().date()),
            "holdout_rows": int(len(data.test)),
        },
        "holdout": {
            "gini": float(row["test_gini"]),
            "auc": float(row["test_auc"]),
            "ks": float(row["test_ks"]),
            "brier": float(row["brier"]),
            "level_error": float(row["level_error"]),
            "calibration_slope": float(row["calib_slope"]),
            "max_quarterly_psi": float(artefacts["psi"]["psi"].max()),
        },
        "code": code,
        "validation_report": "reports/model_validation.md",
    }
    payload = export_champion(card, data.binner, metadata=metadata)

    served = score_frame(champion_from_dict(payload), data.test)
    expected = card.predict_proba(data.woe_test)
    gap = float(np.max(np.abs(served.probability_of_default - expected)))
    if gap > PARITY_TOLERANCE:
        raise RuntimeError(
            f"The exported champion scores the holdout up to {gap:.3g} away "
            f"from the fitted scorecard (tolerance {PARITY_TOLERANCE:g}). The "
            "service would not be scoring the validated model; nothing has "
            "been written."
        )
    return payload, gap


def track(
    config: Config,
    clean: pd.DataFrame,
    leaky: pd.DataFrame,
    artefacts: dict,
    payload: dict,
    parity_gap: float,
    files: list[Path],
):
    """D15: record the run and register the champion it produced."""
    from src.tracking import flatten, metrics_from_table, track_training_run

    params: dict[str, str] = {}
    for section in ("split", "target", "woe", "model", "challenger", "explain"):
        params.update(flatten(section, getattr(config, section)))

    metrics = metrics_from_table(clean)
    metrics.update(metrics_from_table(leaky))
    metrics["champion.max_quarterly_psi"] = float(artefacts["psi"]["psi"].max())
    metrics["champion.serving_parity_max_abs_error"] = parity_gap

    code = payload["metadata"]["code"]
    return track_training_run(
        config.tracking,
        run_name=f"champion {payload['version']}",
        params=params,
        metrics=metrics,
        artefacts=files,
        champion_path=config.serving.artefact_path,
        tags={
            "champion_version": payload["version"],
            "git_commit": code["commit"],
            "git_dirty": code["dirty"],
        },
    )


def main() -> None:
    config = Config.load()
    data = prepare(config)

    print("\nfitting clean models")
    clean, artefacts = evaluate_all(data, config)

    print("\nexporting the champion and checking it scores as fitted")
    from src.tracking import code_version

    payload, parity_gap = export_checked(
        data, clean, artefacts, cutoff=config.split.cutoff_date, code=code_version()
    )
    data_summary = data.summary()
    # Freed before the leaky path loads its own, wider frame. Holding both at
    # once is unnecessary, and this extract is large enough already.
    del data

    print("\nfitting the leaky comparison")
    leaky = evaluate_leaky(config)

    report_path = PROJECT_ROOT / "reports" / "model_validation.md"
    write_report(
        data_summary, clean, leaky, artefacts, report_path,
        config_top=config.explain.top_features,
    )
    write_champion(payload, config.serving.artefact_path)
    print(f"\nwrote {report_path}")
    print(
        f"wrote {config.serving.artefact_path}, version {payload['version']}, "
        f"serving parity {parity_gap:.2e}\n"
    )
    print(clean.to_string(index=False))
    print()
    print(leaky.to_string(index=False))

    if config.tracking.enabled:
        run = track(
            config, clean, leaky, artefacts, payload, parity_gap,
            files=[
                report_path,
                config.leakage.audit_path,
                PROJECT_ROOT / "config" / "config.yaml",
                config.serving.artefact_path,
            ],
        )
        print(
            f"\ntracked run {run.run_id}; registered "
            f"{config.tracking.registered_model} version {run.registered_version}"
        )


if __name__ == "__main__":
    main()
