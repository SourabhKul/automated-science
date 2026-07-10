from core.domain_quarantine import partition_requested_domains


def main():
    stable_domains, quarantined_domains = partition_requested_domains(["ecology", "bz_chem", "fluid"])
    assert stable_domains == ["ecology"]
    assert [item["domain"] for item in quarantined_domains] == ["bz_chem", "fluid"]
    assert all("infinite ABC-SMC metrics" in item["reason"] for item in quarantined_domains)

    stable_domains, quarantined_domains = partition_requested_domains(
        ["ecology", "bz_chem"],
        include_quarantined=True,
    )
    assert stable_domains == ["ecology", "bz_chem"]
    assert quarantined_domains == []

    print("SUCCESS: known unstable domains are quarantined by default and opt-in runnable")


if __name__ == "__main__":
    main()
