"""HTTP interface to the champion.

    uvicorn src.serving.api:create_app --factory

Three endpoints. `POST /score` takes one application and returns the
probability of default, the score, the points per attribute and the principal
reasons (D13). `GET /model` returns what a reviewer needs to know about the
model behind the score: its version, the window it was fitted on, its holdout
metrics and its full points table. `GET /health` is for whatever runs the
container.

The request schema is built from the champion at start-up rather than written
by hand (D14). A hand-written schema is a second copy of the model's inputs,
and the day a retrained champion drops or gains a feature the two disagree
without anything failing. Built from the artefact, the documented contract is
the model's contract: categorical fields list exactly the categories training
saw, and a field is required exactly when training never saw it missing.

Logs carry the request id, status, latency and model version, and never an
applicant's fields. A scoring log is the first place personal data leaks from,
and nothing in operating the service needs the inputs written down.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal, Mapping

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config import Config
from src.serving.artefact import Champion, load_champion
from src.serving.score import (
    InputSpec,
    UnscorableApplication,
    check_reason_texts,
    input_specs,
    score_application,
)

logger = logging.getLogger("credit_risk.serving")

#: Regulation B commentary the reason selection follows. Returned by /model so
#: the method behind every reason is stated, not implied.
REASON_METHOD = (
    "Features on which the applicant scored furthest below the average points "
    "of all training applicants (12 CFR 1002, Supplement I, comment "
    "9(b)(2)-5). At most {n} reasons, per comment 9(b)(2)-1."
)


class AttributeOut(BaseModel):
    feature: str
    bin: str
    points: float


class ReasonOut(BaseModel):
    feature: str
    description: str
    bin: str
    points: float
    reference_points: float
    shortfall: float


class ScoreOut(BaseModel):
    model_version: str
    probability_of_default: float
    score: float
    reasons: list[ReasonOut]
    attributes: list[AttributeOut]


def create_app(
    champion: Champion | None = None,
    reason_texts: Mapping[str, str] | None = None,
    max_reasons: int | None = None,
) -> FastAPI:
    """Build the service. With no arguments, everything comes from config."""
    if champion is None or reason_texts is None or max_reasons is None:
        config = Config.load()
        champion = champion or load_champion(config.serving.artefact_path)
        reason_texts = reason_texts or _read_reason_texts(
            config.serving.reason_texts_path
        )
        max_reasons = max_reasons or config.serving.max_reasons
    check_reason_texts(champion, reason_texts)
    _configure_logging()

    specs = input_specs(champion)
    Application = _application_model(specs, reason_texts)

    app = FastAPI(
        title="Credit risk scoring",
        version=champion.version,
        description=(
            "Probability of default from a WOE logistic-regression scorecard, "
            "with the principal reasons behind every score."
        ),
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000.0
        response.headers["x-request-id"] = request_id
        response.headers["x-model-version"] = champion.version
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.2f model=%s",
            request_id, request.method, request.url.path,
            response.status_code, elapsed, champion.version,
        )
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_version": champion.version}

    @app.get("/model")
    def model() -> dict[str, Any]:
        return {
            "name": champion.name,
            "version": champion.version,
            "base_points": champion.base_points,
            "scaling": dict(champion.scaling),
            "reason_method": REASON_METHOD.format(n=max_reasons),
            "metadata": dict(champion.metadata),
            "inputs": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "required": s.required,
                    "categories": list(s.categories),
                    "feeds": list(s.feeds),
                }
                for s in specs
            ],
            "features": [
                {
                    "name": f.name,
                    "description": reason_texts[f.name],
                    "coefficient": f.coefficient,
                    "reference_points": f.reference_points,
                    "bins": [
                        {"label": b.label, "points": p, "training_rows": b.count}
                        for b, p in zip(f.binning.bins, f.points)
                    ],
                }
                for f in champion.features
            ],
        }

    def score(application) -> ScoreOut:
        try:
            result = score_application(
                champion,
                application.model_dump(),
                reason_texts=reason_texts,
                max_reasons=max_reasons,
            )
        except UnscorableApplication as error:
            raise HTTPException(status_code=422, detail=error.problems) from error
        return ScoreOut(
            model_version=result.model_version,
            probability_of_default=result.probability_of_default,
            score=result.score,
            reasons=[
                ReasonOut(
                    feature=r.feature,
                    description=r.description,
                    bin=r.bin,
                    points=r.points,
                    reference_points=r.reference_points,
                    shortfall=r.shortfall,
                )
                for r in result.reasons
            ],
            attributes=[
                AttributeOut(feature=a.feature, bin=a.bin, points=a.points)
                for a in result.attributes
            ],
        )

    # The request model only exists once the champion is loaded, and under
    # postponed annotations a local class would reach FastAPI as an
    # unresolvable string. The annotation is therefore set on the object.
    score.__annotations__["application"] = Application
    app.post("/score", response_model=ScoreOut)(score)

    return app


def _application_model(
    specs: tuple[InputSpec, ...], reason_texts: Mapping[str, str]
) -> type[BaseModel]:
    """The request schema, one field per model input (D14)."""
    fields: dict[str, Any] = {}
    for spec in specs:
        annotation: Any
        if spec.kind == "categorical":
            annotation = Literal[spec.categories]  # type: ignore[valid-type]
        elif spec.kind == "numeric":
            annotation = float
        elif spec.kind == "date":
            annotation = str
        else:
            annotation = str
        description = "; ".join(
            reason_texts.get(feature, feature) for feature in spec.feeds
        )
        # NaN or infinity would land in an edge bin and score as a real value.
        numeric = {"allow_inf_nan": False} if spec.kind == "numeric" else {}
        if spec.required:
            fields[spec.name] = (
                annotation,
                Field(..., description=description, **numeric),
            )
        else:
            fields[spec.name] = (
                annotation | None,
                Field(None, description=description + " (may be omitted)", **numeric),
            )
    return create_model(
        "Application",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _configure_logging() -> None:
    """Give the service's logger a handler of its own.

    uvicorn configures its own loggers and leaves the root logger without a
    handler, so without this the request log would be silently dropped. The
    logger still propagates, so anything configured above it also sees it.
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _read_reason_texts(path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in raw.items()}
