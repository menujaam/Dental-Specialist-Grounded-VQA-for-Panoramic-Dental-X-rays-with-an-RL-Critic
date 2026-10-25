# Error analysis: v1
Predictions: `preds/dental_lora_3b_image.jsonl`, 3578 test questions.

## 1. Box around a named tooth

| Outcome | Count | Share of present-tooth questions |
| --- | --- | --- |
| one tooth off | 67 | 35.3% |
| right tooth (IoU >= 0.5) | 61 | 32.1% |
| said 'not present' / no box although the tooth exists | 27 | 14.2% |
| missing tooth: correctly said not present | 26 | — |
| right tooth, box too loose/tight (IoU < 0.5) | 23 | 12.1% |
| missing tooth: model drew a box anyway | 15 | — |
| 2+ teeth off, same jaw | 10 | 5.3% |
| box not on any tooth | 2 | 1.1% |

Of 129 wrong boxes on present teeth, 67 are exactly one tooth off (51.9%).

| Tooth type | Correct boxes |
| --- | --- |
| front teeth (incisors) | 17 of 52 (32.7%) |
| wisdom teeth | 1 of 12 (8.3%) |
| premolars | 13 of 46 (28.3%) |
| molars | 19 of 51 (37.3%) |
| canines | 11 of 29 (37.9%) |

## 2. Boxes around findings (diagnosis and 'find all' questions)

| Diagnosis | Expert findings | Found | Missed | Extra model boxes (no expert finding there) |
| --- | --- | --- | --- | --- |
| caries | 700 | 213 (30.4%) | 487 | 559 |
| deep caries | 303 | 47 (15.5%) | 256 | 68 |
| periapical lesion | 48 | 7 (14.6%) | 41 | 8 |
| impacted | 220 | 93 (42.3%) | 127 | 63 |

Found but labelled with the wrong diagnosis: deep caries called caries (37), caries called deep caries (25), caries called impacted (8), periapical lesion called caries (6), deep caries called impacted (5), impacted called caries (4), impacted called deep caries (3), caries called periapical lesion (1), deep caries called periapical lesion (1), periapical lesion called impacted (1).

Found boxes with the wrong tooth number: 86 of 360 (23.9%).

## 3. Duplicate boxes

Answers containing a repeated box: 9.

Repeated copies of a correct box make precision look higher, so the 'after' column is the more honest one.

| Question type | Precision@0.5 before | after removing duplicates | Recall@0.5 (unchanged) |
| --- | --- | --- | --- |
| ground_diagnosis | 34.5% | 34.1% | 21.9% |
| ground_all | 35.9% | 35.9% | 35.9% |
| ground_tooth | 37.4% | 37.4% | 32.1% |

## 4. Diagnosis mix-ups ('What is the finding on tooth X?')

| Expert said | Model said | Count |
| --- | --- | --- |
| caries | caries | 340 |
| impacted | impacted | 98 |
| deep caries | caries | 89 |
| periapical lesion | caries | 19 |
| impacted | caries | 11 |
| caries | impacted | 7 |
| deep caries | impacted | 6 |
| periapical lesion | nothing | 5 |
| deep caries | nothing | 4 |
| caries | nothing | 3 |
| impacted | nothing | 1 |

## 5. 'Is there X?' questions: missed vs invented

| Diagnosis | Correct | Missed (said no, was yes) | Invented (said yes, was no) |
| --- | --- | --- | --- |
| caries | 108 | 0 | 6 |
| deep caries | 63 | 37 | 14 |
| periapical lesion | 95 | 19 | 0 |
| impacted | 92 | 15 | 7 |

Caries depth ('Is the caries on tooth X deep?'):

| Expert | Model said deep | Model said not deep |
| --- | --- | --- |
| deep | 0 | 99 |
| not deep | 0 | 346 |

Deep caries recognised as deep: 0 of 99 (0.0%).

## 6. 'Is tooth X present?' errors by tooth type

| Tooth type | Correct | Said missing, but present | Said present, but missing |
| --- | --- | --- | --- |
| premolars | 57 | 1 | 15 |
| molars | 61 | 3 | 19 |
| wisdom teeth | 63 | 5 | 12 |
| front teeth (incisors) | 56 | 0 | 4 |
| canines | 18 | 0 | 4 |

## 7. Counting and lists

Teeth counted in a quadrant (model minus expert):

| Error | Count |
| --- | --- |
| +0 | 187 |
| +1 | 44 |
| -1 | 133 |
| +2 | 8 |
| -2 | 3 |
| +3 | 2 |
| +4 | 1 |
| -4 | 1 |
| -7 | 1 |

'Which teeth have X?' lists:

| Outcome | Count |
| --- | --- |
| both missed and extra | 157 |
| extra teeth only | 30 |
| missed teeth only | 17 |
| exactly right | 17 |
