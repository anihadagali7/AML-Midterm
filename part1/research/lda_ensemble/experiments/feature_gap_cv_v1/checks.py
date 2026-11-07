"""Focused checks for section parsing and leakage-sensitive feature fitting."""
import json

import numpy as np

import features as m


def check_parsers():
    sample = "Inclusion Criteria:\n - adults\nExclusion Criteria:\n - prior treatment\n"
    split = m.sections(sample)
    assert "adults" in split["inclusion"] and "prior treatment" not in split["inclusion"]
    assert "prior treatment" in split["exclusion"] and "adults" not in split["exclusion"]
    split = m.sections("Exclusion: none\nInclusion: adults\nExclusion: infection")
    assert "none" in split["exclusion"] and "infection" in split["exclusion"]
    assert split["inclusion"] == "adults"
    assert not any(m.sections("DISEASE CHARACTERISTICS:\n - unknown format").values())
    nested = json.dumps(["['C4A.0', 'C34.2', 'C34.2']", "['D3A.8']"])
    full = m.categories(nested, "icd_full_category")
    assert full == {"code:C4A.0", "code:C34.2", "code:D3A.8", "category:C4A", "category:C34", "category:D3A"}
    assert m.categories(nested, "icd_category") == {"category:C4A", "category:C34", "category:D3A"}
    assert m.categories(float("nan"), "drug_targets") == set()
    assert m.categories(" ERBB2 ; EGFR; ERBB2 ", "drug_targets") == {"erbb2", "egfr"}
    assert m.categories("['Lung Cancer', ' lung  cancer ']", "diseases") == {"lung cancer"}


def check_boundary(packet, fit, query):
    checks = []
    assert not set(fit.ID) & set(query.ID)
    for name in m.BLOCKS:
        block = m.Block(name).fit(fit)
        before = block.transform(fit).copy()
        changed = query.copy()
        changed.loc[:, "criteria"] = "Inclusion Criteria: ONLYQUERYTOKEN\nExclusion Criteria: QUERYONLY"
        block.transform(changed)
        np.testing.assert_array_equal(before, block.transform(fit))
        assert block.fitting_ids == fit.ID.tolist()
        assert not any("onlyquerytoken" in value or "queryonly" in value for value in block.feature_names)
        # Predicting a singleton must match its prediction in a larger query batch.
        all_scores, all_meta, audit = m.fit_addition(packet, fit, query, name)
        one, one_meta, single_audit = m.fit_addition(packet, fit, query.iloc[:1], name)
        np.testing.assert_allclose(all_scores[:1], one, atol=1e-10, rtol=0)
        np.testing.assert_allclose(all_meta[:1], one_meta, atol=1e-10, rtol=0)
        assert audit["feature_names_sha256"] == single_audit["feature_names_sha256"]
        assert audit["multiplier"] == single_audit["multiplier"]
        checks.append(dict(block=name, query_batch_invariance=True, fitting_transform_unchanged=True,
                           query_only_token_excluded=True, dimensions=len(block.feature_names)))
    return dict(parser_checks_passed=True, feature_boundary_checks=checks)


if __name__ == "__main__":
    check_parsers()
    print("Structured parsing checks passed.")
