| Training data | Framework | Accuracy | Defect recall | Precision | Missed defects | False alarms | Size (MB) | ms / image |
|---|---|---|---|---|---|---|---|---|
| Real photos | keras | 99.58% | 99.34% | 100.00% | 3 | 0 | 22.97 | 158.94 |
| Real photos | pytorch | 99.16% | 99.34% | 99.34% | 3 | 3 | 44.79 | 3.01 |
| CAD only | keras | 62.66% | 94.26% | 63.92% | 26 | 241 | 22.97 | 162.18 |
| CAD only | pytorch | 63.36% | 100.00% | 63.36% | 0 | 262 | 44.79 | 3.16 |
| CAD + 10% real | keras | 84.06% | 99.78% | 80.00% | 1 | 113 | 22.97 | 164.01 |
| CAD + 10% real | pytorch | 98.46% | 99.56% | 98.04% | 2 | 9 | 44.79 | 3.62 |
