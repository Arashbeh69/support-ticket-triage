# Limitations and unsupported claims

- The task is English-only, banking-specific, and limited to short single-query
  classification with no conversation history.
- BANKING77 provides intent labels, not priority, agent, resolution, operational
  queue, customer outcome, or business-impact ground truth.
- Source documentation does not fully describe collection, annotation,
  agreement, privacy review, population, or demographic coverage.
- Fourteen credential-value candidates were detected and redacted, but rules
  cannot guarantee complete sensitive-data detection or correct every flag.
- Duplicate and near-duplicate definitions are heuristic. The rejected 0.92
  threshold demonstrated that ordinary intent similarity can resemble leakage.
- The official test is balanced at 40 examples per intent and may not resemble
  production prevalence, unknown intents, adversarial inputs, or distribution
  shift.
- Confidence is not correctness. Temperature scaling improves a measured
  calibration objective but does not create an open-set guarantee.
- Abstention and domain-mismatch rules are portfolio design proposals, not
  production-validated banking policy.
- High-risk review rules reduce automation by design but do not prove that all
  harmful routing errors are captured.
- There is no production deployment, live monitoring, human-operations study,
  security review, fairness claim, causal estimate, cost-saving estimate, or
  demonstrated business impact.
- The service is not financial, security, fraud, or identity-verification advice
  and must not automate account action.
