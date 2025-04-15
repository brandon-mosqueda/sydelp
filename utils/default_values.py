from utils.utils import as_name

def fill_with_defaults(params: dict) -> dict:
    # These are the mandatory params
    attack: str = as_name(params['attack'])
    dataset: str = as_name(params['dataset'])
    protocol: str = as_name(params['protocol'])

    is_test: bool = params.get('is_test', False)
    defaults: dict = {}

    if attack == 'label_flipping':
        # These are always assigned if not present for this attack
        if "attack_success_rate" not in params['metrics']:
            params['metrics'].append("attack_success_rate")

        if "label_recall" not in params['metrics']:
            params['metrics'].append("label_recall")

        if dataset == 'sms_spam':
            defaults['source_label'] = 1
            defaults['target_label'] = 0
        elif dataset == 'mnist':
            defaults['source_label'] = 1
            defaults['target_label'] = 9
        else:
            raise ValueError(f"Dataset '{params['dataset']}' not valid")

        if is_test:
            params['label_flip_malicious_num'] = 2
    elif attack == "random":
        defaults['attack_mean'] = 3
        defaults['attack_sd'] = 2

        if is_test:
            params['random_malicious_num'] = 2
    elif attack == "sign_flipping":
        defaults['attack_scale_factor'] = 3

        if is_test:
            params['sign_flip_malicious_num'] = 2

    if protocol == "sydelp":
        defaults['difficulty_alpha'] = 1
        defaults['momentum_coeff'] = 0.1
        defaults['models_per_iteration'] = 40 if not is_test else 8
        defaults['expected_malicious_num'] = 18 if not is_test else 2

        defaults['is_worst_case'] = False
        defaults['computing_power'] = 0
    elif protocol == "sybilwall":
        defaults['graph_type'] = "random_regular"
        defaults["degree"] = 8 if not is_test else 4
        defaults["distant_propagation_relevance"] = 0.8
        defaults["confidence"] = 1
    elif protocol == "mab-fl":
        if dataset == "mnist":
            defaults["warm_up_iterations"] = 8 if not is_test else 2
            defaults["mab_alpha"] = -0.1
            defaults["miu"] = 0.1
            defaults["c_max"] = 0.7
            defaults["c_min"] = 0.3
            defaults["pca_components"] = 0.95
        elif dataset == "sms_spam":
            defaults["warm_up_iterations"] = 8
            defaults["mab_alpha"] = 0.1
            defaults["miu"] = 0.1
            defaults["c_max"] = 0.93
            defaults["c_min"] = 0.93
            defaults["pca_components"] = 0.95

    for key, value in defaults.items():
        params[key] = params.get(key, value)

    return params
