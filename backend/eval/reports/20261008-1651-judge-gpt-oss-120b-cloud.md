# Experiment B judged: RAG triad (2026-10-08 16:51)

30 questions per configuration · judge `gpt-oss:120b-cloud` · rows: `20261008-1628-e2e-rows.jsonl`, `20261008-1644-e2e-gemma4-31b-cloud-rows.jsonl`

Context relevance = share of the chunks given to the writer that help answer the question (judged whenever the writer got chunks, even if it then declined; agentic mode gives the writer up to 10 chunks, traditional 5). Faithfulness = share of the answer's claims supported by those chunks. Answer relevance = does the answer address the question (yes 1, partly 0.5, no 0). Faithfulness and answer relevance are judged only on answers given. n = rows scored. One run per configuration: differences of 1-2 questions may be run-to-run noise. Scores come from an LLM judge and can be wrong in both directions; none of them checks the answer against the golden key facts.

## Table 1: RAG triad

|  | gemma4:e2b traditional | gemma4:e2b agentic | gemma4:31b-cloud traditional | gemma4:31b-cloud agentic |
|---|---|---|---|---|
| Context relevance, mean | 0.27 (n=30) | 0.22 (n=23) | 0.24 (n=30) | 0.38 (n=14) |
| Faithfulness, mean | 0.93 (n=26) | 0.94 (n=23) | 1.00 (n=17) | 1.00 (n=14) |
| Answer relevance, mean | 0.69 (n=26) | 0.61 (n=23) | 1.00 (n=17) | 0.86 (n=14) |

Answers given / declined (not found, out of scope, timeout, error) / judge calls invalid twice: gemma4:e2b traditional: 26 / 4 / 0 · gemma4:e2b agentic: 23 / 7 / 0 · gemma4:31b-cloud traditional: 17 / 13 / 0 · gemma4:31b-cloud agentic: 14 / 16 / 0

## Table 2: Per question

CR = context relevance · F = faithfulness · AR = answer relevance · – = not judged

| ID | Question | gemma4:e2b traditional | gemma4:e2b agentic | gemma4:31b-cloud traditional | gemma4:31b-cloud agentic |
|---|---|---|---|---|---|
| S01 | Within how many hours must an NCII enti… | CR 0.20 · F 1.00 · AR 1.00 | CR – · said not found | CR 0.20 · F 1.00 · AR 1.00 | CR – · said not found |
| S02 | How often must an NCII entity conduct a… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.33 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.33 · F 1.00 · AR 1.00 |
| S03 | Through which system must cyber securit… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.17 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR – · said not found |
| S04 | Which sectors are national critical inf… | CR 0.00 · F 0.00 · AR 0.00 | CR 0.00 · F 0.00 · AR 0.00 | CR 0.00 · said not found | CR – · said not found |
| S05 | What is the penalty for providing a cyb… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.14 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 |
| S06 | How long must a licensed cyber security… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.09 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 |
| S07 | Does the Cyber Security Act 2024 apply … | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.40 · F 1.00 · AR 1.00 |
| S08 | How does the Act define a cyber securit… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 |
| S09 | If an NCII entity makes a material chan… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 |
| S10 | What is the penalty if an NCII entity d… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.10 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 |
| M01 | It is 5pm on Friday and ransomware has … | CR 0.40 · F 1.00 · AR 1.00 | CR – · said not found | CR 0.40 · F 1.00 · AR 1.00 | CR 0.57 · F 1.00 · AR 1.00 |
| M02 | Is a private hospital automatically bou… | CR 0.20 · said not found | CR – · said not found | CR 0.20 · said not found | CR 1.00 · F 1.00 · AR 1.00 |
| M03 | We are an NCII entity. What recurring s… | CR 0.20 · F 1.00 · AR 1.00 | CR 0.30 · F 1.00 · AR 0.50 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 0.50 |
| M04 | Which is punished more heavily: not rep… | CR 0.40 · F 0.80 · AR 1.00 | CR 0.20 · F 1.00 · AR 0.00 | CR 0.40 · said not found | CR 0.14 · F 1.00 · AR 0.50 |
| M05 | We think our sector's security rulebook… | CR 0.60 · F 1.00 · AR 0.00 | CR 0.10 · F 1.00 · AR 0.50 | CR 0.40 · said not found | CR 0.50 · F 1.00 · AR 0.50 |
| M06 | The regulator told us to fix our system… | CR 0.40 · F 1.00 · AR 0.00 | CR 0.30 · F 1.00 · AR 0.50 | CR 0.20 · F 1.00 · AR 1.00 | CR – · said not found |
| M07 | Our cyber security service provider los… | CR 0.40 · F 1.00 · AR 0.00 | CR 0.33 · F 1.00 · AR 0.00 | CR 0.20 · said not found | CR – · said not found |
| M08 | What is the largest compound payment th… | CR 0.40 · F 0.50 · AR 0.50 | CR 0.20 · F 1.00 · AR 0.00 | CR 0.40 · F 1.00 · AR 1.00 | CR – · said not found |
| M09 | Who decides whether my company counts a… | CR 0.20 · F 1.00 · AR 0.50 | CR 0.43 · F 1.00 · AR 1.00 | CR 0.20 · F 1.00 · AR 1.00 | CR 0.50 · F 1.00 · AR 1.00 |
| M10 | Our IT vendor offers security testing s… | CR 0.60 · F 1.00 · AR 1.00 | CR 0.50 · F 1.00 · AR 0.50 | CR 0.60 · F 1.00 · AR 1.00 | CR 0.67 · F 1.00 · AR 0.50 |
| N01 | What does the Personal Data Protection … | CR 0.00 · said not found | CR 0.00 · F 1.00 · AR 0.00 | CR 0.00 · said not found | CR – · refused: out of scope |
| N02 | What is the penalty for unauthorised ac… | CR 0.00 · F 1.00 · AR 0.00 | CR – · refused: out of scope | CR 0.00 · said not found | CR – · refused: out of scope |
| N03 | Who is the current Chief Executive of t… | CR 0.00 · F 1.00 · AR 0.00 | CR 0.00 · F 1.00 · AR 0.00 | CR 0.00 · said not found | CR – · refused: out of scope |
| N04 | How much is the fee for a cyber securit… | CR 0.00 · F 1.00 · AR 0.00 | CR 0.00 · F 0.67 · AR 0.00 | CR 0.00 · said not found | CR – · said not found |
| N05 | How should I configure my firewall to b… | CR 0.00 · said not found | CR – · refused: out of scope | CR 0.00 · said not found | CR – · refused: out of scope |
| A01 | When does the law come into force? | CR 0.60 · F 1.00 · AR 1.00 | CR 0.40 · F 1.00 · AR 1.00 | CR 0.60 · F 1.00 · AR 1.00 | CR – · said not found |
| A02 | What is the penalty? | CR 0.20 · F 1.00 · AR 1.00 | CR – · said not found | CR 0.20 · said not found | CR – · said not found |
| A03 | Do I have to report it? | CR 0.60 · F 1.00 · AR 1.00 | CR 0.50 · F 1.00 · AR 1.00 | CR 0.60 · said not found | CR – · said not found |
| A04 | How long do we have? | CR 0.40 · said not found | CR 0.33 · F 1.00 · AR 1.00 | CR 0.20 · said not found | CR – · said not found |
| A05 | Are we covered by the Act? | CR 0.80 · F 1.00 · AR 1.00 | CR – · said not found | CR 0.60 · F 1.00 · AR 1.00 | CR – · said not found |
