"""Domain-neutral masking profiles and display names.

Profiles define policy; detectors define capability. The default profile masks
common direct and quasi-identifiers across document types. HIPAA Safe Harbor is
available as an explicit US-healthcare profile rather than shaping universal
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

# Common recognizer-backed entity types. DATE is handled separately so calendar
# validation can reject measurements and malformed dates.
PATTERN_ENTITIES = [
    "PERSON",  # 1  names
    "LOCATION",  # 2  geographic subdivisions
    "PHONE_NUMBER",  # 4  telephone / 5 fax
    "EMAIL_ADDRESS",  # 6  email
    "US_SSN",  # 7  social security number
    "MEDICAL_LICENSE",  # 10 certificate / license numbers
    "US_DRIVER_LICENSE",  # 12 licence plate / driver ID
    "URL",  # 14 web URLs
    "IP_ADDRESS",  # 15 IP addresses
    "US_PASSPORT",
    "US_ITIN",
    "US_BANK_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "CRYPTO",
]

# Contextual types need document labels, geometry, or visual detectors.
CONTEXTUAL_ENTITIES = [
    "GENERIC_ID",
    "MEDICAL_RECORD_NUMBER",
    "ACCOUNT_NUMBER",
    "HEALTH_PLAN_ID",
    "CLAIM_NUMBER",
    "DATE",
    "AGE",
    "FACE",
    "SIGNATURE",
]

GENERAL_PII = PATTERN_ENTITIES + CONTEXTUAL_ENTITIES

# Kept as a public API alias for existing callers. Selecting this profile is an
# explicit policy decision; it is no longer the package default.
SAFE_HARBOR = [
    *PATTERN_ENTITIES,
    "GENERIC_ID",
    "MEDICAL_RECORD_NUMBER",
    "ACCOUNT_NUMBER",
    "HEALTH_PLAN_ID",
    "CLAIM_NUMBER",
    "DATE",
    "AGE",
    "FACE",
    "SIGNATURE",
]


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    entities: tuple[str, ...]
    min_masked_age: int
    mask_professional_people: bool = True
    mask_organizations: bool = False


PROFILES = {
    "general": PolicyProfile("general", tuple(GENERAL_PII), 0),
    "hipaa-safe-harbor": PolicyProfile(
        "hipaa-safe-harbor", tuple(SAFE_HARBOR), 90, False
    ),
}


def profile(name: str = "general") -> PolicyProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown policy profile {name!r}; choose from {sorted(PROFILES)}"
        ) from exc

# Full calendar dates are masked; standalone years are retained.
DATE_POLICY = "mask_full_dates_keep_years"

# Short display names keep replacement tags legible inside compact fields.
SHORT_NAMES = {
    "PERSON": "NAME",
    "LOCATION": "ADDR",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "ACCOUNT_NUMBER": "ACCT",
    "HEALTH_PLAN_ID": "PLAN",
    "CLAIM_NUMBER": "CLAIM",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "US_DRIVER_LICENSE": "DL",
    "US_PASSPORT": "PPT",
    "MEDICAL_LICENSE": "LIC",
    "US_BANK_NUMBER": "BANK",
    "CREDIT_CARD": "CARD",
    "IP_ADDRESS": "IP",
    "US_ITIN": "ITIN",
    "IBAN_CODE": "IBAN",
    "AGE": "AGE",
    "FACE": "FACE",
    "SIGNATURE": "SIG",
    "ORGANIZATION": "ORG",
    "GENERIC_ID": "ID",
}


# Libraries must not silently exempt real-world identities. Callers may supply
# an allowlist explicitly for their own policy and document set.
DEFAULT_ALLOWLIST: tuple[str, ...] = ()
