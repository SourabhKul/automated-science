from __future__ import annotations

from typing import Any


KNOWN_QUARANTINED_DOMAINS: dict[str, str] = {
    "bz_chem": "Seed evaluation currently returns infinite ABC-SMC metrics under the gaussian_weighted sandbox path.",
    "fluid": "Seed evaluation currently returns infinite ABC-SMC metrics under the gaussian_weighted sandbox path.",
}


def quarantine_reason(domain: str) -> str | None:
    return KNOWN_QUARANTINED_DOMAINS.get(domain)


def partition_requested_domains(
    domains: list[str],
    *,
    include_quarantined: bool = False,
) -> tuple[list[str], list[dict[str, str]]]:
    stable_domains: list[str] = []
    quarantined_domains: list[dict[str, str]] = []
    for domain in domains:
        reason = quarantine_reason(domain)
        if reason and not include_quarantined:
            quarantined_domains.append({
                "domain": domain,
                "reason": reason,
            })
            continue
        stable_domains.append(domain)
    return stable_domains, quarantined_domains


def quarantine_domains_from_records(records: list[Any] | None) -> list[str]:
    names: list[str] = []
    for record in records or []:
        if isinstance(record, str):
            names.append(record)
            continue
        if isinstance(record, dict) and record.get("domain"):
            names.append(str(record["domain"]))
    return names
