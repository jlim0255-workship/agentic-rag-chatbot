# Experiment B judged: faithfulness, relevance, correctness (2026-10-04 16:24)

30 questions per configuration · judge `gpt-oss:120b-cloud` · rows: `20261004-1617-e2e-rows.jsonl`

Correctness = share of the golden key facts the answer covers (partly = half), judged against `answer_notes`; a not-in-docs question scores 1 if declined, 0 if answered. Faithfulness = share of the answer's claims supported by the chunks given to the writer. Relevance = does it address the question (yes 1, partly 0.5, no 0). One run per configuration: differences of 1-2 questions may be run-to-run noise. Scores come from an LLM judge and can be wrong in both directions.

## Table 1: Headline

|  | gemma4:e2b traditional | gemma4:e2b agentic |
|---|---|---|
| Correct (score >= 0.75), all questions | 8/30 (27%) | 10/30 (33%) |
| Correctness, mean | 0.47 (n=30) | 0.44 (n=29) |
| Faithfulness, mean (answers with claims) | 0.95 (n=26) | 0.85 (n=18) |
| Fully faithful answers | 24/26 (92%) | 11/18 (61%) |
| Answer relevance, mean (answers given) | 0.69 (n=26) | 0.67 (n=18) |
| Answers contradicting a key fact | 3 | 0 |
| Right sections given to the writer (FULL) | 13/23 (57%) | 12/23 (52%) |
| Multi-hop questions: FULL | 3/10 (30%) | 4/10 (40%) |
| Not-in-docs questions declined (OK) | 2/5 (40%) | 3/5 (60%) |
| Answerable questions wrongly declined | 1/20 (5%) | 5/20 (25%) |
| Citations pointing at a needed section | 23/30 (77%) | 15/27 (56%) |
| Latency p50 / p95 | 6.2 s / 21.7 s | 16.3 s / 51.5 s |
| LLM calls per question (mean) | 1.0 | 7.2 |
| Timeouts / errors | 0 / 0 | 0 / 0 |
| Judge calls invalid twice (unjudged) | 0 | 1 |

## Table 2: Correct by category

| Category | gemma4:e2b traditional | gemma4:e2b agentic |
|---|---|---|
| single_hop (10) | 4/10 (40%) | 6/10 (60%) |
| multi_hop (10) | 1/10 (10%) | 1/10 (10%) |
| not_in_corpus (5) | 2/5 (40%) | 3/5 (60%) |
| ambiguous (5) | 1/5 (20%) | 0/5 (0%) |

## Table 3: Why answers were wrong (questions with known answer sections)

Correct = score >= 0.75; faithful = every claim supported (or no claims). Order of checks: correct? then faithful? then were all needed sections given to the writer?

| Diagnosis | gemma4:e2b traditional | gemma4:e2b agentic |
|---|---|---|
| Success | 6/23 (26%) | 7/23 (30%) |
| Correct but not grounded | 0/23 (0%) | 0/23 (0%) |
| Hallucination | 2/23 (9%) | 6/23 (26%) |
| Writer missed facts it had | 8/23 (35%) | 2/23 (9%) |
| Retrieval failure | 7/23 (30%) | 8/23 (35%) |

## Table 4: Per question

C = correctness · F = faithfulness · R = relevance · – = not judged

| ID | Question | gemma4:e2b traditional | gemma4:e2b agentic |
|---|---|---|---|
| S01 | Within how many hours must an NCII enti… | C 0.50 · F 1.00 · R 1.00 | C 0.00 · said not found |
| S02 | How often must an NCII entity conduct a… | C 1.00 · F 1.00 · R 1.00 | C 1.00 · F 1.00 · R 1.00 |
| S03 | Through which system must cyber securit… | C 0.75 · F 1.00 · R 1.00 | C 1.00 · F 1.00 · R 1.00 |
| S04 | Which sectors are national critical inf… | C 0.00 · F 0.00 · R 0.00 | C – · F 0.00 · R 0.00 |
| S05 | What is the penalty for providing a cyb… | C 0.50 · F 1.00 · R 1.00 | C 0.75 · F 1.00 · R 1.00 |
| S06 | How long must a licensed cyber security… | C 0.50 · F 1.00 · R 1.00 | C 0.50 · F 1.00 · R 1.00 |
| S07 | Does the Cyber Security Act 2024 apply … | C 1.00 · F 1.00 · R 1.00 | C 1.00 · F 1.00 · R 1.00 |
| S08 | How does the Act define a cyber securit… | C 1.00 · F 1.00 · R 1.00 | C 1.00 · F 1.00 · R 1.00 |
| S09 | If an NCII entity makes a material chan… | C 0.50 · F 1.00 · R 1.00 | C 1.00 · F 1.00 · R 1.00 |
| S10 | What is the penalty if an NCII entity d… | C 0.50 · F 1.00 · R 1.00 | C 0.50 · F 0.75 · R 1.00 |
| M01 | It is 5pm on Friday and ransomware has … | C 0.50 · F 1.00 · R 1.00 | C 0.00 · said not found |
| M02 | Is a private hospital automatically bou… | C 0.00 · said not found | C 0.00 · said not found |
| M03 | We are an NCII entity. What recurring s… | C 0.88 · F 1.00 · R 1.00 | C 0.62 · F 0.88 · R 0.50 |
| M04 | Which is punished more heavily: not rep… | C 0.25 · F 0.80 · R 1.00 | C 0.50 · F 1.00 · R 0.50 |
| M05 | We think our sector's security rulebook… | C 0.25 · F 1.00 · R 0.00 | C 0.00 · said not found |
| M06 | The regulator told us to fix our system… | C 0.67 · F 1.00 · R 0.00 | C 0.33 · F 0.92 · R 0.00 |
| M07 | Our cyber security service provider los… | C 0.00 · F 1.00 · R 0.00 | C 0.00 · said not found |
| M08 | What is the largest compound payment th… | C 0.50 · F 1.00 · R 0.50 | C 0.00 · F 0.94 · R 0.00 |
| M09 | Who decides whether my company counts a… | C 0.50 · F 1.00 · R 0.50 | C 1.00 · F 1.00 · R 1.00 |
| M10 | Our IT vendor offers security testing s… | C 0.67 · F 1.00 · R 1.00 | C 0.33 · F 0.91 · R 0.50 |
| N01 | What does the Personal Data Protection … | C 1.00 · said not found | C 1.00 · said not found |
| N02 | What is the penalty for unauthorised ac… | C 0.00 · F 1.00 · R 0.00 | C 1.00 · refused: out of scope |
| N03 | Who is the current Chief Executive of t… | C 0.00 · F 1.00 · R 0.00 | C 0.00 · F 1.00 · R 0.00 |
| N04 | How much is the fee for a cyber securit… | C 0.00 · F 1.00 · R 0.00 | C 0.00 · F 1.00 · R 0.50 |
| N05 | How should I configure my firewall to b… | C 1.00 · said not found | C 1.00 · refused: out of scope |
| A01 | When does the law come into force? | C 1.00 · F 1.00 · R 1.00 | C 0.00 · said not found |
| A02 | What is the penalty? | C 0.00 · F 1.00 · R 1.00 | C 0.00 · said not found |
| A03 | Do I have to report it? | C 0.50 · F 1.00 · R 1.00 | C 0.00 · said not found |
| A04 | How long do we have? | C 0.00 · said not found | C 0.25 · F 0.00 · R 1.00 |
| A05 | Are we covered by the Act? | C 0.00 · F 1.00 · R 1.00 | C 0.00 · said not found |
