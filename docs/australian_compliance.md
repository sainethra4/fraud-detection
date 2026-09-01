# Australian Regulatory and Governance Documentation

## Scope and status

This is an operational compliance checklist for the fraud-detection prototype. It is not legal advice and must be reviewed by the insurer's legal, privacy, risk, and claims teams before production use.

## 1. Privacy Act 1988 and Australian Privacy Principles

The dataset contains personal information, including age, gender, marital status, income, location indicators, and claim history. Before production use, establish whether the organisation is subject to the Privacy Act 1988 (Cth) and document the purpose and lawful basis for every use of personal information.

| Control | Required operational practice |
| --- | --- |
| Purpose and minimisation | Use only fields reasonably necessary for fraud detection; regularly review whether each feature remains necessary. |
| Notice and transparency | Explain in the privacy policy and collection notice that claims information may be analysed to detect and investigate fraud, including relevant disclosures and complaint channels. |
| Data quality | Validate, correct, and monitor data quality before using it to make or support a claims decision. Treat `*`, blanks, and malformed dates as missing, as this project does. |
| Security | Restrict access by role; encrypt data in transit and at rest; use secure secrets management; maintain audit logs; review third-party/cloud access; test incident response. |
| Retention | Apply an approved retention schedule. Destroy or de-identify data when it is no longer needed unless retention is legally required. |
| Access and correction | Maintain a process to locate relevant personal information, handle access/correction requests, and correct source data where appropriate. |

The Australian Privacy Principles require reasonable steps for data quality and security; APP 11 covers protection from misuse, interference, loss, and unauthorised access, modification, or disclosure. APP 12 covers access requests. See the [OAIC APP quick reference](https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-quick-reference), [OAIC APP 3 guidance](https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/chapter-3-app-3-collection-of-solicited-personal-information), and [OAIC APP 11 guidance](https://www.oaic.gov.au/privacy/australian-privacy-principles/australian-privacy-principles-guidelines/chapter-11-app-11-security-of-personal-information).

## 2. Insurance Contracts Act 1984

Claims handling must be carried out consistently with the duty of utmost good faith. The model's score is a triage input only: it cannot be the sole basis for rejecting, reducing, delaying, or settling a claim. An appropriately trained claims professional must assess evidence, apply policy terms, and document the decision. See the [Insurance Contracts Act 1984](https://www.legislation.gov.au/C2004A02944/2024-03-01/2024-03-01/text/original/pdf).

## 3. Human review, fairness, and explainability

- Preserve a human-in-the-loop workflow: `investigate = True` sends a claim for review, not an automated adverse decision.
- Record the model version, score, threshold, reviewer, evidence considered, and final decision for each reviewed claim.
- Test performance and false-positive rates across relevant groups where lawful, statistically sound, and appropriate. Escalate material disparities to governance owners.
- Give investigators a plain-language explanation of score drivers and prohibit use of protected or irrelevant attributes as the sole reason for an adverse decision.
- Maintain an appeal, complaint, and correction pathway for customers.

## 4. Production release checklist

- [ ] Privacy impact assessment and legal review completed.
- [ ] Approved data inventory, collection notice, and retention schedule exist.
- [ ] Role-based access, encryption, audit logging, and incident response are tested.
- [ ] Model validation covers accuracy, recall, specificity, calibration, drift, and fairness.
- [ ] Threshold and review capacity have an approved business owner.
- [ ] Claims staff have documented review procedures and escalation paths.
- [ ] Monitoring, retraining, rollback, and periodic independent review are scheduled.

