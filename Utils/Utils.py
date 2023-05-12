def normalize_meshId(x):
    return x.upper().replace('MESH:', '')


def is_valid_relation(drug_id, disease_id, test_set) -> bool:
    try:
        return test_set[(
            normalize_meshId(drug_id),
            normalize_meshId(disease_id),
        )]
    except KeyError:
        return False
