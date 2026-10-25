# Census — which clips enter the experiment (gates only; set rule in docs/plan_experiment_v1.md)

Camera gate: triangulated split, median held-out reprojection <= 0.3 patch. Support: band points >= 30, half-split <= 0.05.

| clip | pool | target | Module 1 | admission | camera gate: median px (n) -> outcome | band n / half-split / ratio [CI] | support | CENSUS |
|---|---|---|---|---|---|---|---|---|
| bear | davis | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| bike-packing | davis | human | ok | ok | 0.20 (219) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 129 / 0.023 / 0.861 [0.7748150079032712, 1.0227652500181266] | pass | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol))** |
| blackswan | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| bmx-bumps | davis | human | ok | ok | 0.25 (346) -> pass | 232 / 0.007 / 0.952 [0.9353676373649955, 0.9739353068526803] | pass | **PASS** |
| bmx-trees | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| boat | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| boxing-fisheye | davis | human | ok | ok | 0.46 (312) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 79 / 0.056 / 0.752 [0.6959743065141341, 0.8802813490721059] | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol)); support surface undetermined** |
| breakdance | davis | human | ok | ok | 0.18 (161) -> pass | 161 / 0.003 / 0.953 [0.8909256634014695, 0.9852869962333406] | pass | **PASS** |
| breakdance-flare | davis | human | ok | ok | 0.18 (156) -> pass | 137 / 0.026 / 0.989 [0.9761521518417239, 1.0209688877749798] | pass | **PASS** |
| bus | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| camel | davis | human | ok | ok | 0.42 (293) -> pass | 87 / 0.001 / 0.986 [0.9746903587977486, 1.003550160150936] | pass | **PASS** |
| car-roundabout | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| car-turn | davis | vehicle | ok | ok | 0.25 (361) -> undetermined (band gauge: undetermined (n=15<30)) | 15 / - / -  | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (n=15<30)); support surface undetermined** |
| cat-girl | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| classic-car | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| color-run | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| cows | davis | human | ok | ok | 0.30 (126) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 72 / 0.084 / 0.947 [0.9019298910223025, 1.0804449907564442] | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol)); support surface undetermined** |
| crossing | davis | human | ok | ok | 0.13 (226) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 57 / 0.043 / 2.100 [1.7966483605413068, 2.8463721228366023] | pass | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol))** |
| dance-jump | davis | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| dance-twirl | davis | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| dancing | davis | human | ok | ok | 0.54 (285) -> pass | 131 / 0.011 / 0.276 [0.2562620075914528, 0.2872726861943171] | pass | **PASS** |
| disc-jockey | davis | human | ok | ok | 0.54 (105) -> pass | 55 / 0.013 / 0.991 [0.9798165799789923, 0.9959694478069554] | pass | **PASS** |
| dog | davis | human | ok | ok | - (-) -> undetermined (camera C unsolved on some frames: too few static tracks there) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (camera C unsolved on some frames: too few static tracks there)** |
| dog-gooses | davis | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| dogs-jump | davis | human | ok | ok | 0.38 (343) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 44 / 0.158 / 6.008 [4.9816601572277115, 6.688424689631374] | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol)); support surface undetermined** |
| dogs-scale | davis | human | ok | ok | 0.55 (347) -> pass | 273 / 0.014 / 1.034 [1.0139756582817543, 1.0475976494021808] | pass | **PASS** |
| drift-chicane | davis | vehicle | ok | ok | 0.14 (272) -> undetermined (band gauge: undetermined (half-split or interval wider than tol)) | 150 / 0.227 / 1.053 [0.886231959484092, 1.4542562444504852] | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (half-split or interval wider than tol)); support surface undetermined** |
| drift-straight | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| drift-turn | davis | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| drone | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **UNDETERMINED: Module 1 stage C crashed — ValueError: zero-size array to reduction operation minimum which has no identity** |
| elephant | davis | human | ok | ok | 0.26 (278) -> pass | 244 / 0.016 / 1.036 [1.0210587023624353, 1.0921197138562662] | pass | **PASS** |
| flamingo | davis | human | ok | ok | 0.30 (45) -> undetermined (band gauge: undetermined (n=29<30)) | 29 / - / -  | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (n=29<30)); support surface undetermined** |
| goat | davis | human | ok | ok | 0.34 (317) -> pass | 202 / 0.029 / 0.934 [0.9145468695183179, 0.9509879129588267] | pass | **PASS** |
| gold-fish | davis | human | ok | ok | 0.18 (221) -> pass | 173 / 0.002 / 0.996 [0.991725212838192, 1.0001202990264437] | pass | **PASS** |
| hike | davis | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| hockey | davis | human | ok | ok | 0.27 (240) -> pass | 193 / 0.004 / 1.021 [1.0072605337319334, 1.0286641900795341] | pass | **PASS** |
| horsejump-high | davis | human | ok | ok | 0.41 (373) -> pass | 155 / 0.001 / 1.033 [1.0063614622408397, 1.0522505618717495] | pass | **PASS** |
| horsejump-low | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| kid-football | davis | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| kite-walk | davis | human | ok | ok | 0.21 (91) -> pass | 65 / 0.016 / 1.016 [0.9963031712090563, 1.0305906021731692] | pass | **PASS** |
| soapbox | davis | vehicle | ok | ok | 0.22 (22) -> undetermined (band gauge: undetermined (n=1<30)) | 1 / - / -  | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (n=1<30)); support surface undetermined** |
| 000000000001.5.002 | miradata | human | ok | ok | - (-) -> UNDETERMINED: camera instruments did not run (Reading (reviewer, pre-stated): inside the first window camera C should coincide with G to about the) | - / - / -  | not evaluated (the camera did not reach the gate) | **UNDETERMINED: camera instruments did not run (Reading (reviewer, pre-stated): inside the first window camera C should coincide with G to about the)** |
| 000000000035.1.003 | miradata | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| 000000000064.3.002 | miradata | human | ok | ok | 0.23 (350) -> pass | 285 / 0.006 / 1.001 [0.9950556335514955, 1.0065885248095472] | pass | **PASS** |
| 000000000100.0.003 | miradata | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| 000000000119.5.007 | miradata | human | ok | ok | 0.23 (312) -> pass | 161 / 0.001 / 0.848 [0.8446025519205049, 0.8545453823251408] | pass | **PASS** |
| 000000000130.0.015 | miradata | human | ok | ok | 0.42 (778) -> pass | 602 / 0.004 / 0.998 [0.9925028628961682, 1.0047250188302659] | pass | **PASS** |
| 000000000191.6.037 | miradata | human | ok | ok | 0.34 (427) -> pass | 212 / 0.003 / 0.968 [0.965712367501271, 0.9703181571839316] | pass | **PASS** |
| 000000000191.6.038 | miradata | human | ok | ok | 0.19 (227) -> undetermined (band gauge: undetermined (n=22<30)) | 22 / - / -  | bridge undefined: no support surface (band points or half-split) | **FAIL: camera undetermined (band gauge: undetermined (n=22<30)); support surface undetermined** |
| 000000000200.1.006 | miradata | human | ok | ok | 0.28 (483) -> pass | 309 / 0.004 / 1.035 [1.028022232245152, 1.0400243017035489] | pass | **PASS** |
| 000000000201.1.008 | miradata | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| 000000000217.0.006 | miradata | vehicle | ok | ok | 0.48 (599) -> pass | 156 / 0.025 / 0.989 [0.978071081786899, 1.0007554145573077] | pass | **PASS** |
| 000000000221.7.011 | miradata | human | ok | ok | 0.30 (565) -> pass | 337 / 0.005 / 0.966 [0.9629781714428933, 0.9701802260162372] | pass | **PASS** |
| 000000000222.4.003 | miradata | human | ok | ok | 0.23 (378) -> pass | 292 / 0.001 / 0.995 [0.9917653749924713, 0.9989127386835448] | pass | **PASS** |
| 000000000230.2.013 | miradata | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| 000000000252.1.018 | miradata | vehicle | ok | ok | 0.48 (858) -> pass | 463 / 0.005 / 1.008 [1.0046389639046422, 1.0119403022689963] | pass | **PASS** |
| 000000000252.2.007 | miradata | human | ok | ok | 0.43 (662) -> pass | 439 / 0.000 / 0.994 [0.9883998355271029, 1.0007445345678094] | pass | **PASS** |
| 000000000256.3.010 | miradata | human | ok | ok | 0.32 (581) -> pass | 274 / 0.007 / 0.980 [0.9750910876512591, 0.9883546098137851] | pass | **PASS** |
| 000000000265.2.005 | miradata | human | ok | ok | - (-) -> undetermined (too few static points for the camera solve) | - / - / -  | not evaluated (the camera did not reach the gate) | **FAIL: camera undetermined (too few static points for the camera solve)** |
| 000000000265.4.002 | miradata | human | ok | ok | 0.29 (572) -> pass | 328 / 0.002 / 0.996 [0.9935271150223829, 0.9987855894618735] | pass | **PASS** |
| 000000000284.0.006 | miradata | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2)** |
| 000000000299.1.005 | miradata | vehicle | ok | ok | 0.48 (465) -> pass | 211 / 0.027 / 0.980 [0.9486347328561561, 1.0169320320251811] | pass | **PASS** |
| 000000000362.2.005 | miradata | human | ok | ok | 0.29 (298) -> pass | 206 / 0.001 / 1.001 [0.9980509652190686, 1.005316180936446] | pass | **PASS** |
| 000000000370.1.009 | miradata | human | ok | ok | 0.21 (378) -> pass | 246 / 0.002 / 0.954 [0.9486830078372113, 0.9584618671474233] | pass | **PASS** |
| 000000000391.0.016 | miradata | human | ok | ok | 0.19 (381) -> pass | 211 / 0.013 / 0.977 [0.9676510001772121, 0.982161670123965] | pass | **PASS** |
| 000000000396.3.002 | miradata | vehicle | ok | ok | 0.50 (577) -> pass | 364 / 0.015 / 1.011 [1.0019655375092313, 1.0338918769285605] | pass | **PASS** |
| 000000000404.1.006 | miradata | human | ok | ok | 0.78 (573) -> pass | 254 / 0.002 / 1.017 [1.0092856167438382, 1.027174779545016] | pass | **PASS** |
| 000000000418.0.005 | miradata | vehicle | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |
| 000000000431.5.009 | miradata | human | ok | ok | 0.31 (567) -> pass | 460 / 0.001 / 1.005 [1.0031442432607027, 1.0100369426983924] | pass | **PASS** |
| 000000002376.4.008 | miradata | human | FAIL | - | - (-) -> - | - / - / -  | - | **FAIL: Module 1 stage A gate refused it (A-G2b)** |

Pass rate: 32 / 70. A FAIL row is reported, never adjusted; the experiment set is taken from the PASS rows by the pre-written rule.