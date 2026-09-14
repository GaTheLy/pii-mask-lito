"""Agent-assisted masking built from small, independently checkable steps.

Language models are useful for interpreting unfamiliar layouts and labels, but
their coordinates are not reliable enough to place masks. The model therefore
describes fields and decisions while deterministic code resolves the returned
value text back to OCR tokens and page geometry:

    what kind of document is this        -> agent      (semantic)
    which fields exist, and whose        -> agent      (semantic)
    mask or keep, and why                -> agent      (judgement)
    which token holds a field's value    -> resolver   (deterministic)
    where that token sits on the page    -> OCR        (deterministic)
    which tag it gets                    -> registry   (deterministic)
    did anything leak                    -> auditor    (deterministic)

An agent never controls a coordinate, and rule detections remain authoritative.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .detect import SpatialContextDetector, _TOKEN_EDGE
from .model import Span, TokenText

# Rule results are authoritative. Agent output may add detections but cannot
# remove them, because a semantic misclassification must not expose a value.
UNVETOABLE = {"US_SSN", "EMAIL_ADDRESS", "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE"}

# Free-text type names a model returns, mapped onto the pipeline's vocabulary.
TYPE_ALIASES = {
    "date": "DATE",
    "date of birth": "DATE",
    "dob": "DATE",
    "birthdate": "DATE",
    "person": "PERSON",
    "name": "PERSON",
    "full name": "PERSON",
    "employee name": "PERSON",
    "customer name": "PERSON",
    "address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "zip": "LOCATION",
    "zip code": "LOCATION",
    "postal code": "LOCATION",
    "location": "LOCATION",
    "ssn": "US_SSN",
    "social security number": "US_SSN",
    "phone": "PHONE_NUMBER",
    "phone number": "PHONE_NUMBER",
    "telephone": "PHONE_NUMBER",
    "fax": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "record id": "GENERIC_ID",
    "record number": "GENERIC_ID",
    "employee id": "GENERIC_ID",
    "customer id": "GENERIC_ID",
    "user id": "GENERIC_ID",
    "reference id": "GENERIC_ID",
    "account": "ACCOUNT_NUMBER",
    "account number": "ACCOUNT_NUMBER",
    "identifier": "GENERIC_ID",
    "identification": "GENERIC_ID",
    "id number": "GENERIC_ID",
}


def canonical_type(raw: str) -> str | None:
    key = re.sub(r"[^a-z' ]", " ", (raw or "").casefold()).strip()
    key = " ".join(key.split())
    if not key:
        return None
    if key.upper().replace(" ", "_") in set(TYPE_ALIASES.values()):
        return key.upper().replace(" ", "_")
    return TYPE_ALIASES.get(key)


@dataclass
class Field:
    """One field an agent found, before it has been grounded to a token."""

    label: str
    value: str = ""
    owner: str = "unknown"
    entity: str = ""
    hint_index: int | None = None
    decision: str = ""
    reason: str = ""


@dataclass
class Trace:
    """What each step did, so an agentic run stays inspectable."""

    steps: list[dict] = field(default_factory=list)

    def add(self, agent: str, seconds: float, detail: dict) -> None:
        self.steps.append({"agent": agent, "seconds": round(seconds, 1), **detail})


# Longest edge, in pixels, of the page image sent to the model.
#
# OCR benefits from a high-resolution source. The model receives the OCR token
# list separately and needs the image only for layout context, so a smaller
# raster saves vision tokens without changing mask geometry.
VLM_MAX_EDGE = 1400


def _encode(image, max_edge: int) -> str:
    """Downscale to `max_edge` and return base64 PNG.

    Shared by every transport: the size limit is a property of what the model
    needs from the image, not of who is hosting it.
    """
    scale = max_edge / max(image.size)
    if scale < 1:
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _first_json(body: str) -> dict:
    """The first JSON object in a reply, however the model wrapped it."""
    match = re.search(r"\{.*\}", body, re.S)
    if not match:
        raise ValueError(f"no JSON in model reply: {body[:200]!r}")
    return json.loads(match.group(0))


def _post(url: str, payload: dict, timeout: int, headers: dict | None = None,
          attempts: int = 4) -> dict:
    """POST JSON, retrying the failures that are worth retrying.

    Retries exist for the hosted path. A local server either answers or is down,
    but an API rate-limits, and a 100-page document is 300 calls: without this a
    single 429 two hundred pages in costs that page its recall silently, which
    is the one failure mode this tool is built to not have. 5xx and connection
    resets get the same treatment; 4xx other than 429 is a bad request and
    retrying it just wastes the quota.
    """
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == attempts - 1:
                detail = exc.read()[:300].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {exc.code} from {url.split('?')[0]}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class Ollama:
    """Local model transport; document content stays on the configured host."""

    def __init__(
        self,
        model: str = "gemma4:31b",
        host: str = "http://localhost:11434",
        timeout: int = 180,
        num_ctx: int = 8192,
        max_edge: int = VLM_MAX_EDGE,
    ):
        self.model, self.host, self.timeout, self.num_ctx = model, host.rstrip("/"), timeout, num_ctx
        self.max_edge = max_edge

    def ask(self, prompt: str, image=None) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if image is not None:
            payload["images"] = [_encode(image, self.max_edge)]
        body = _post(f"{self.host}/api/generate", payload, self.timeout, attempts=1)["response"]
        return _first_json(body)


# Hosted transports send page images and OCR text to another service. Keep this
# runtime warning visible so operators can apply their own privacy and data-
# processing requirements before enabling one.
HOSTED_NOTICE = (
    "  !! %s sends page images and OCR text to %s.\n"
    "     Confirm that this destination is approved for the document's data class.\n"
    "     Everything except --agents still runs entirely locally."
)


class Gemini:
    """Google Generative Language API transport.

    Same contract as Ollama -- prompt in, one JSON object out -- so the agents
    above cannot tell which one they are holding. Hosted inference is useful
    when the selected local model does not fit the available hardware.

    What it does not remove is the division of labour. This model still never
    returns a coordinate; it reads fields and attributes owners, and the
    resolver and OCR put those values on the page. A hosted model is better at
    reading, not better at pointing.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: str | None = None,
        timeout: int = 180,
        max_edge: int = VLM_MAX_EDGE,
        endpoint: str | None = None,
        quiet: bool = False,
    ):
        # Never a CLI flag: an argument lands in shell history and in `ps`.
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not self.api_key:
            raise RuntimeError(
                "no API key: set GEMINI_API_KEY (or GOOGLE_API_KEY) to use a hosted "
                "model. Get one at https://aistudio.google.com/apikey"
            )
        self.model, self.timeout, self.max_edge = model, timeout, max_edge
        self.endpoint = (endpoint or self.ENDPOINT).rstrip("/")
        if not quiet:
            host = self.endpoint.split("/")[2]
            print(HOSTED_NOTICE % (model, host), file=sys.stderr, flush=True)

    def ask(self, prompt: str, image=None) -> dict:
        parts: list[dict] = [{"text": prompt}]
        if image is not None:
            parts.append(
                {"inline_data": {"mime_type": "image/png",
                                 "data": _encode(image, self.max_edge)}}
            )
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            # responseMimeType is the API's own JSON mode, and it is stricter
            # than asking for JSON in the prompt: the reply is parseable or the
            # request fails, rather than arriving as prose with an object in it.
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        reply = _post(
            f"{self.endpoint}/{self.model}:generateContent",
            payload,
            self.timeout,
            headers={"x-goog-api-key": self.api_key},
        )
        candidates = reply.get("candidates") or []
        if not candidates:
            # Some safety filters return no candidate instead of an error.
            raise ValueError(f"no candidate returned: {str(reply)[:200]}")
        chunks = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
        return _first_json("".join(chunks))


class OpenAICompatible:
    """Any /v1/chat/completions endpoint: vLLM, SGLang, OpenRouter, Azure, LM Studio.

    One class rather than one per vendor, because the wire format is the same
    and the only thing that varies is the base URL and which env var holds the
    key. That is what makes the choice of host a deployment decision instead of
    a code change -- including self-hosting a large VLM on a GPU box, which is
    the option that gets the local path's privacy and the hosted path's ceiling.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        key_env: str = "OPENAI_API_KEY",
        timeout: int = 180,
        max_edge: int = VLM_MAX_EDGE,
        quiet: bool = False,
    ):
        self.model, self.timeout, self.max_edge = model, timeout, max_edge
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(key_env, "")
        if not quiet and not self.base_url.startswith(("http://localhost", "http://127.")):
            print(HOSTED_NOTICE % (model, self.base_url.split("/")[2]),
                  file=sys.stderr, flush=True)

    def ask(self, prompt: str, image=None) -> dict:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image is not None:
            url = f"data:image/png;base64,{_encode(image, self.max_edge)}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        reply = _post(f"{self.base_url}/chat/completions", payload, self.timeout, headers)
        choices = reply.get("choices") or []
        if not choices:
            raise ValueError(f"no choice returned: {str(reply)[:200]}")
        return _first_json(choices[0].get("message", {}).get("content") or "")


# Where a model name that carries no explicit provider is served from. Prefix
# rather than an exact list on purpose: a list goes stale the week a vendor
# ships a new version, and the failure mode is this tool refusing a model that
# works.
_PREFIXES = (
    ("gemini", "gemini"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude-", "anthropic"),
)

_OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "vllm": ("http://localhost:8000/v1", "VLLM_API_KEY"),
}


def transport(spec: str, host: str = "http://localhost:11434", **kwargs):
    """Build a transport from one string, so the host is a runtime choice.

    Accepts "provider:model" explicitly, or a bare model name whose provider is
    inferred from its prefix. Anything unrecognised is Ollama, which keeps the
    local path the default in the one way that matters: you have to name a
    hosted provider to get one.

        gemma4:31b                       -> Ollama          (local)
        qwen2.5vl:7b                     -> Ollama          (local)
        gemini-3.8-flash                 -> Gemini          (hosted)
        openrouter:qwen/qwen2.5-vl-72b   -> OpenAI-compatible
        vllm:Qwen/Qwen2.5-VL-32B         -> OpenAI-compatible, localhost:8000
        openai-compatible:MODEL@https://my-gpu-box:8000/v1
    """
    provider, _, model = spec.partition(":")
    provider = provider.casefold()

    # "gemma4:31b" and "qwen2.5vl:7b" are Ollama tags, not provider prefixes,
    # so an unknown left-hand side means the whole string is a model name.
    if provider not in {"ollama", "gemini", "openai-compatible"} | set(_OPENAI_COMPATIBLE):
        provider, model = "", spec
    if not provider:
        provider = next((p for prefix, p in _PREFIXES if spec.startswith(prefix)), "ollama")

    if provider == "gemini":
        return Gemini(model=model or "gemini-3.8-flash", **kwargs)
    if provider == "ollama":
        return Ollama(model=model or spec, host=host, **kwargs)
    if provider == "openai-compatible":
        model, _, base = model.partition("@")
        if not base:
            raise ValueError(
                "openai-compatible needs a base URL: openai-compatible:MODEL@https://host/v1"
            )
        return OpenAICompatible(model=model, base_url=base, **kwargs)
    base, key_env = _OPENAI_COMPATIBLE[provider]
    return OpenAICompatible(model=model, base_url=base, key_env=key_env, **kwargs)


# --------------------------------------------------------------------------
# Agents. Each has one job and a small, checkable output.
# --------------------------------------------------------------------------

CLASSIFY = """You are looking at one page of a document that may contain \
personal or sensitive information.

Identify the document type and the blocks that may contain information about \
identifiable people.

Reply with JSON only:
{"doc_type":"<e.g. application, invoice, correspondence, identity form, \
spreadsheet, report, unknown>",
 "regions":["<short name of each distinct block on the page>"],
 "has_personal_data": true|false}"""


READ_FIELDS = """This is a page from a document (%(doc_type)s).

These are the text tokens on the page, numbered:
%(tokens)s

List every labelled field on the page. For each one give the field's printed \
label, the value you can read for it, and WHOSE information it is.

owner must be exactly one of: subject, employee, customer, applicant, signer, \
sender, recipient, professional, organization, other.

Ownership describes context only. A person's name remains a personal identifier \
whether that person is an employee, customer, professional, signer, sender, or \
recipient. Do not treat a person's job or role as permission to expose them.

Reply with JSON only:
{"fields":[{"label":"<printed label>","value":"<value text>",\
"owner":"subject|employee|customer|applicant|signer|sender|recipient|\
professional|organization|other",\
"type":"<name|date|ssn|address|phone|email|account number|identifier|\
amount|code|other>"}]}"""


ADJUDICATE = """You are masking personal and sensitive information in a \
%(doc_type)s.

Mask identifiers tied to a natural person: names; detailed addresses; dates of \
birth and other person-specific dates; phone and fax numbers; email addresses; \
government, financial, account, certificate, licence, vehicle, device, network, \
biometric, and other unique identifiers. Apply this consistently to every \
person regardless of occupation or relationship to the document.

Keep information that is not itself personal, such as ordinary organization \
names, generic product or transaction codes, quantities, and monetary amounts. \
When ownership or sensitivity is uncertain, choose mask.

Here are the fields found on the page:
%(fields)s

For each field decide whether it must be masked.

Reply with JSON only:
{"decisions":[{"label":"<the field label, copied exactly>","action":"mask|keep",\
"reason":"<a few words>"}]}"""


class Classifier:
    """Step 1 -- what am I looking at?

    Cheap and small. Its answer conditions the prompts of every later step, so
    the reader can use the document type while extracting fields.
    """

    name = "classifier"

    def __init__(self, model: Ollama):
        self.model = model

    def run(self, image) -> dict:
        try:
            reply = self.model.ask(CLASSIFY, image)
        except Exception as exc:  # noqa: BLE001 - degrade, never block masking
            return {"doc_type": "unknown", "regions": [], "error": str(exc)}
        if not isinstance(reply, dict):
            return {"doc_type": "unknown", "regions": [], "has_personal_data": True}
        regions = reply.get("regions")
        return {
            "doc_type": str(reply.get("doc_type") or "unknown")[:80],
            "regions": [str(r)[:40] for r in regions][:20] if isinstance(regions, list) else [],
            "has_personal_data": bool(
                reply.get("has_personal_data", reply.get("has_patient_data", True))
            ),
        }


class FieldReader:
    """Step 2 -- which fields exist, and whose information is each one?

    This is the step that replaces a hand-written label list. It discovers the
    labels actually printed on the page, and attributes each to an owner.
    """

    name = "field_reader"

    def __init__(self, model: Ollama):
        self.model = model

    def run(self, image, tt: TokenText, doc_type: str) -> list[Field]:
        # A dense page yields 300+ tokens; the listing alone can crowd out the
        # image. Punctuation and single marks carry no field information.
        useful = [(i, t.text) for i, t in enumerate(tt.tokens) if len(t.text.strip()) > 1]
        listing = "\n".join(f"{i}: {text}" for i, text in useful[:400])
        prompt = READ_FIELDS % {"doc_type": doc_type, "tokens": listing}
        try:
            reply = self.model.ask(prompt, image)
        except Exception:  # noqa: BLE001
            return []
        return _coerce_fields(reply)


def _coerce_fields(reply) -> list[Field]:
    """Turn whatever the model returned into Fields, or nothing.

    Model output is untrusted input. Asked for a list of objects, a 7B model
    handed back a list of bare strings on a dense page -- and the resulting
    AttributeError killed a ten-page run outright, ten minutes in. Shape is
    checked here, not assumed.
    """
    raw = reply.get("fields") if isinstance(reply, dict) else reply
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return []
    fields = []
    for item in raw[:80]:
        if not isinstance(item, dict):
            continue  # a bare string carries no value or owner; nothing to ground
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        fields.append(
            Field(
                label=label,
                value=str(item.get("value") or "").strip(),
                owner=str(item.get("owner") or "unknown").strip().casefold(),
                entity=canonical_type(str(item.get("type") or "")) or "",
            )
        )
    return fields


class Adjudicator:
    """Step 3 -- mask or keep, with a reason.

    Separated from reading on purpose. A 7B model asked to extract *and* decide
    in one pass does both worse, and keeping the decision isolated means the
    reason for every mask is recorded and reviewable.
    """

    name = "adjudicator"

    def __init__(self, model: Ollama):
        self.model = model

    def run(self, fields: list[Field], doc_type: str) -> list[Field]:
        if not fields:
            return fields
        listing = "\n".join(
            f"- label={f.label!r} value={f.value!r} owner={f.owner} type={f.entity or 'unknown'}"
            for f in fields
        )
        prompt = ADJUDICATE % {"doc_type": doc_type, "fields": listing}
        decisions: dict[str, tuple[str, str]] = {}
        try:
            reply = self.model.ask(prompt)
            for raw in (reply.get("decisions") if isinstance(reply, dict) else None) or []:
                if not isinstance(raw, dict):
                    continue
                label = str(raw.get("label") or "").strip().casefold()
                action = str(raw.get("action") or "").strip().casefold()
                if label and action in {"mask", "keep"}:
                    decisions[label] = (action, str(raw.get("reason") or "")[:80])
        except Exception:  # noqa: BLE001 - fall through to the owner heuristic
            pass

        for f in fields:
            action, reason = decisions.get(f.label.casefold(), ("", ""))
            if not action:
                # Model output is advisory. Missing or malformed decisions use
                # the conservative default for an extracted identifying field.
                action = "mask"
                reason = "conservative default"
            f.decision, f.reason = action, reason
        return fields


class Auditor:
    """Step 6 -- look at the finished page and report anything still readable.

    Deterministic read-back already guarantees no *detected* value survives.
    This catches the other half: sensitive text no detector ever proposed.
    """

    name = "auditor"
    PROMPT = """This page has been de-identified. Masked values are covered by \
boxes containing indexed tags.

Look for any remaining personally identifying information about any natural \
person that is still readable: names, addresses, dates of birth, phone numbers, \
email addresses, government identifiers, or record/account identifiers.

Ignore generic codes, quantities, and monetary amounts that do not identify a \
person. Do not ignore a name merely because the person has a professional role.

Reply with JSON only: {"remaining":["<exact text still readable>"]}"""

    def __init__(self, model: Ollama):
        self.model = model

    def run(self, image) -> list[str]:
        try:
            reply = self.model.ask(self.PROMPT, image)
        except Exception:  # noqa: BLE001
            return []
        return [str(v)[:80] for v in (reply.get("remaining") or [])][:20]


# --------------------------------------------------------------------------
# Step 4: grounding. Deterministic, because this is where the model fails.
# --------------------------------------------------------------------------


_MONEY = re.compile(r"^\$?\d{1,3}(,\d{3})*(\.\d{2})?$|^\$?\d+\.\d{2}$")

# Identifier shape for the agentic path: at least four characters,
# alphanumeric, and containing a digit. It is deliberately permissive because
# identifier formats vary by domain. This only narrows tokens within a field the
# agent already selected; it does not decide whether a field is sensitive.
_ID_SHAPE = re.compile(r"^(?=.*\d)[A-Z0-9][A-Z0-9\-/]{3,}$", re.I)


def resolve_value(tt: TokenText, value: str, entity: str = "") -> list[int]:
    """Locate the tokens holding a value the agent read off the page.

    The model supplies value text while this function supplies position by
    ordinary text search. That separation tolerates synonymous labels and
    avoids trusting model-generated coordinates or token indices.
    """
    words = [w for w in re.findall(r"[0-9A-Za-z@.]+", value or "") if w]
    if not words:
        return []
    # Words are joined by a permissive separator rather than plain whitespace.
    # Labels and values may be separated by punctuation that OCR emits as its
    # own token, so allow a short run of non-alphanumeric separators.
    for cut in range(len(words), 0, -1):
        pattern = r"[^0-9A-Za-z]{0,8}".join(re.escape(w) for w in words[:cut])
        match = re.search(pattern, tt.text, re.I)
        if not match:
            continue  # values come back truncated; retry with a shorter prefix
        return _narrow(tt, tt.tokens_for(match.start(), match.end()), entity)
    return []


def _narrow(tt: TokenText, indices: list[int], entity: str) -> list[int]:
    """Drop tokens that are the field's printed label rather than its value.

    Models sometimes include part of a printed label in the value. For entity
    types with a definite shape, keep only tokens that match that shape so the
    document's static caption is not masked.
    """
    if not indices or not entity:
        return indices
    kept = [i for i in indices if _shape_score(tt.tokens[i].text.strip(_TOKEN_EDGE), entity) > 0]
    if kept:
        return kept
    # Nothing here is shaped like the entity the agent named, so the anchor is
    # wrong. Returning the tokens anyway would mask a label; drop it instead and
    # let the rule detector cover this field.
    return [] if entity in _DEFINITE_SHAPE else indices


_DEFINITE_SHAPE = {
    "MEDICAL_RECORD_NUMBER",
    "ACCOUNT_NUMBER",
    "HEALTH_PLAN_ID",
    "CLAIM_NUMBER",
    "GENERIC_ID",
    "US_SSN",
    "DATE",
    "EMAIL_ADDRESS",
}


def _shape_score(text: str, entity: str) -> float:
    """Does this token look like the entity the agent said it was?"""
    if not entity:
        return 0.0
    if entity in {
        "MEDICAL_RECORD_NUMBER", "ACCOUNT_NUMBER", "HEALTH_PLAN_ID",
        "CLAIM_NUMBER", "GENERIC_ID",
    }:
        return 0.5 if _ID_SHAPE.match(text) else -0.5
    if entity == "DATE":
        return 0.5 if re.search(r"\d{2}[/-]?\d{2}", text) else -0.5
    if entity == "US_SSN":
        return 0.5 if re.match(r"\d{3}-?\d{2}-?\d{4}$", text) else -0.5
    if entity == "EMAIL_ADDRESS":
        return 0.5 if "@" in text else -0.5
    if entity == "PERSON":
        return 0.5 if text[:1].isalpha() and not text.isdigit() else -0.5
    if entity == "LOCATION":
        return 0.3 if not text.isdigit() or len(text) in (4, 5) else 0.0
    return 0.0


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


class AgenticDetector:
    """Runs the agent pipeline and reconciles it with the rule detector.

    Reconciliation is a union: agents may add fields whose labels or layouts
    the rules did not anticipate, but they cannot withdraw a rule detection.
    This keeps semantic model mistakes fail-closed.
    """

    def __init__(self, rules, model: Ollama | None = None, audit: bool = False,
                 verbose: bool = True):
        self.rules = rules
        self.model = model or Ollama()
        self.spatial = SpatialContextDetector()
        self.classifier = Classifier(self.model)
        self.reader = FieldReader(self.model)
        self.adjudicator = Adjudicator(self.model)
        self.auditor = Auditor(self.model) if audit else None
        self.trace = Trace()
        self.page = 0
        self.verbose = verbose

    def _note(self, message: str, since: float) -> None:
        if self.verbose:
            print(f"       {message}  ({time.time() - since:.0f}s)", flush=True)

    def detect(self, image, tt: TokenText) -> list[Span]:
        rule_spans = self.rules.detect(tt)
        if image is None:
            return rule_spans
        try:
            return self._agent_pass(image, tt, rule_spans)
        except Exception as exc:  # noqa: BLE001
            # Agents are additive. A model that returns an unexpected shape, or
            # a host that goes away mid-document, must cost this page its extra
            # recall -- not the whole run's output.
            self.trace.add("agents", 0.0, {"failed": type(exc).__name__})
            return rule_spans

    def _agent_pass(self, image, tt: TokenText, rule_spans: list[Span]) -> list[Span]:
        self.page += 1
        t0 = time.time()
        info = self.classifier.run(image)
        self._note(f"page {self.page}: {info['doc_type'][:40]}", t0)
        self.trace.add(self.classifier.name, time.time() - t0, {"doc_type": info["doc_type"]})

        t0 = time.time()
        fields = self.reader.run(image, tt, info["doc_type"])
        self._note(f"page {self.page}: read {len(fields)} fields", t0)
        self.trace.add(self.reader.name, time.time() - t0, {"fields": len(fields)})

        t0 = time.time()
        fields = self.adjudicator.run(fields, info["doc_type"])
        masked = [f for f in fields if f.decision == "mask"]
        self._note(f"page {self.page}: mask {len(masked)}, keep {len(fields)-len(masked)}", t0)
        self.trace.add(
            self.adjudicator.name,
            time.time() - t0,
            {"mask": len(masked), "keep": len(fields) - len(masked)},
        )

        t0 = time.time()
        agent_spans, keep_tokens = self._ground(tt, fields)
        self.trace.add(
            "resolver",
            time.time() - t0,
            {"grounded": len(agent_spans), "kept_tokens": len(keep_tokens)},
        )

        return self._reconcile(rule_spans, agent_spans, keep_tokens)

    def _ground(self, tt: TokenText, fields: list[Field]):
        """Turn masking decisions into spans.

        The returned empty set preserves the historical internal interface;
        model decisions no longer veto rule detections.
        """
        spans, keep_tokens = [], set()
        for f in fields:
            indices = resolve_value(tt, f.value, f.entity)
            if not indices:
                continue
            if f.decision == "keep":
                continue
            entity = f.entity or "ACCOUNT_NUMBER"
            for i in indices:
                text = tt.tokens[i].text.strip(_TOKEN_EDGE)
                if not text or _MONEY.match(text):
                    continue
                start = tt.offsets[i][0] + tt.tokens[i].text.find(text)
                spans.append(
                    Span(
                        entity=entity,
                        start=start,
                        end=start + len(text),
                        score=0.7,
                        text=text,
                        tokens=[i],
                        source="agent",
                    )
                )
        return spans, keep_tokens

    def _reconcile(self, rule_spans, agent_spans, keep_tokens):
        from .model import merge_spans

        # `keep_tokens` is accepted for compatibility with older callers and
        # traces, but model output is additive and cannot expose a rule hit.
        return merge_spans(rule_spans + agent_spans)


def build(rules, model_name: str = "gemma4:31b", host: str = "http://localhost:11434",
          audit: bool = False, verbose: bool = True) -> AgenticDetector:
    return AgenticDetector(rules, transport(model_name, host=host), audit=audit,
                           verbose=verbose)
