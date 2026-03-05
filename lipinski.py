import logging
from typing import Tuple, Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

# Lipinski Rule of Five thresholds
LIPINSKI_RULES = {
    "MW": {"limit": 500, "operator": "<=", "full_name": "Molecular Weight ≤ 500 Da"},
    "LogP": {"limit": 5.0, "operator": "<=", "full_name": "LogP ≤ 5"},
    "HBD": {"limit": 5, "operator": "<=", "full_name": "H-Bond Donors ≤ 5"},
    "HBA": {"limit": 10, "operator": "<=", "full_name": "H-Bond Acceptors ≤ 10"},
}


def compute_lipinski_descriptors(smiles: str) -> Dict[str, float]:
    """
    Compute Lipinski descriptors for a molecule.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary with descriptor values
        Keys: MW, LogP, HBD, HBA

    Raises:
        ValueError: If SMILES is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        descriptors = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
        }

        logger.debug(f"Computed descriptors for {smiles}: {descriptors}")
        return descriptors

    except Exception as e:
        logger.error(f"Error computing descriptors for {smiles}: {e}")
        raise


def check_lipinski_rule(descriptor_key: str, value: float) -> bool:
    """
    Check if a single descriptor passes Lipinski rule.

    Args:
        descriptor_key: One of 'MW', 'LogP', 'HBD', 'HBA'
        value: Descriptor value

    Returns:
        True if passes, False otherwise

    Raises:
        KeyError: If descriptor_key not in LIPINSKI_RULES
    """
    if descriptor_key not in LIPINSKI_RULES:
        raise KeyError(f"Unknown descriptor: {descriptor_key}")

    rule = LIPINSKI_RULES[descriptor_key]
    limit = rule["limit"]
    operator = rule["operator"]

    if operator == "<=":
        return value <= limit
    raise ValueError(f"Unknown operator: {operator}")


def evaluate_lipinski(smiles: str) -> Tuple[bool, int, Dict]:
    """
    Evaluate Lipinski Rule of Five compliance for a molecule.

    Args:
        smiles: SMILES string

    Returns:
        Tuple of:
            - passes: bool, True if all rules pass
            - violations: int, number of violated rules
            - detailed: dict with full evaluation details

    Raises:
        ValueError: If SMILES is invalid
    """
    # Compute descriptors
    descriptors = compute_lipinski_descriptors(smiles)

    # Evaluate each rule
    violations = 0
    rule_results = {}

    for key, value in descriptors.items():
        passed = check_lipinski_rule(key, value)
        rule_results[key] = {
            "value": value,
            "limit": LIPINSKI_RULES[key]["limit"],
            "full_name": LIPINSKI_RULES[key]["full_name"],
            "passed": passed,
        }
        if not passed:
            violations += 1

    passes = violations == 0

    # Build detailed report
    detailed = {
        "smiles": smiles,
        "passes": passes,
        "violations": violations,
        "rules": rule_results,
    }

    logger.info(
        f"Lipinski evaluation for {smiles}: {violations} violation(s). "
        f"Passes: {passes}"
    )

    return passes, violations, detailed


def lipinski_summary(smiles: str) -> Dict[str, bool]:
    """
    Get simple boolean compliance for each Lipinski rule.

    Convenience function for quick rule checking.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary mapping rule names to compliance (True/False)

    Raises:
        ValueError: If SMILES is invalid
    """
    passes, violations, detailed = evaluate_lipinski(smiles)
    summary = {}
    for key, rule_data in detailed["rules"].items():
        summary[rule_data["full_name"]] = rule_data["passed"]
    return summary


def batch_lipinski(smiles_list: List[str]) -> List[Dict]:
    """
    Evaluate Lipinski compliance for multiple molecules.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of dictionaries, each containing:
            - smiles: input SMILES
            - passes: bool
            - violations: int
            - valid: bool (False if SMILES invalid or evaluation failed)
    """
    results = []

    for smiles in smiles_list:
        try:
            passes, violations, detailed = evaluate_lipinski(smiles)
            results.append(
                {
                    "smiles": smiles,
                    "passes": passes,
                    "violations": violations,
                    "valid": True,
                }
            )
        except (ValueError, Exception) as e:
            logger.warning(f"Skipping SMILES {smiles}: {e}")
            results.append(
                {
                    "smiles": smiles,
                    "passes": False,
                    "violations": -1,
                    "valid": False,
                }
            )

    return results


def get_violation_messages(smiles: str) -> List[str]:
    """
    Get human-readable messages for each Lipinski violation.

    Args:
        smiles: SMILES string

    Returns:
        List of violation messages (empty if no violations)

    Raises:
        ValueError: If SMILES is invalid
    """
    passes, violations, detailed = evaluate_lipinski(smiles)
    messages = []

    for key, rule_data in detailed["rules"].items():
        if not rule_data["passed"]:
            msg = (
                f"{rule_data['full_name']}: "
                f"{rule_data['value']:.2f} > {rule_data['limit']}"
            )
            messages.append(msg)

    return messages
