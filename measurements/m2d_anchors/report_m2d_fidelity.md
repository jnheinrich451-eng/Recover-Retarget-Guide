# m2_d step 6 — fidelity on the retained candidates (recorded, not gated)

| clip | cat | seed | obj IoU | bg rho | E_loc | E_loc vs rendered centroid | log(A_gen/A_obj) | det |
|---|---|---|---|---|---|---|---|---|
| 35.1.003 | vehicle | 0 | 0.911 | +0.789 | 0.0029 | 0.0028 | -0.058 | 0.38 |
| 35.1.003 | vehicle | 1 | 0.812 | +0.677 | 0.0097 | 0.0103 | -0.149 | 0.42 |
| 35.1.003 | vehicle | 2 | 0.924 | +0.759 | 0.0049 | 0.0045 | -0.064 | 0.44 |
| 35.1.003 | vehicle | 3 | 0.898 | +0.694 | 0.0046 | 0.0046 | -0.082 | 0.46 |
| 84.0.006 | vehicle | 0 | 0.900 | +0.974 | 0.0009 | 0.0009 | +0.033 | 0.32 |
| 84.0.006 | vehicle | 1 | 0.835 | +0.972 | 0.0115 | 0.0112 | -0.081 | 0.29 |
| 84.0.006 | vehicle | 2 | 0.827 | +0.967 | 0.0090 | 0.0088 | +0.061 | 0.42 |
| 84.0.006 | vehicle | 3 | 0.845 | +0.969 | 0.0114 | 0.0112 | -0.068 | 0.30 |
| 65.2.005 | vehicle | 0 | 0.971 | +0.839 | 0.0017 | 0.0016 | +0.008 | 0.48 |
| 65.2.005 | vehicle | 1 | 0.922 | +0.857 | 0.0083 | 0.0083 | -0.071 | 0.41 |
| 65.2.005 | vehicle | 2 | 0.963 | +0.862 | 0.0008 | 0.0009 | -0.019 | 0.46 |
| 65.2.005 | vehicle | 3 | 0.908 | +0.886 | 0.0095 | 0.0095 | -0.076 | 0.40 |
| 65.2.005 | human | 0 | 0.626 | +0.915 | 0.0075 | 0.0070 | +0.468 | 0.30 |
| 65.2.005 | human | 1 | 0.677 | +0.922 | 0.0021 | 0.0017 | +0.389 | 0.32 |
| 65.2.005 | human | 2 | 0.677 | +0.941 | 0.0073 | 0.0068 | +0.385 | 0.37 |
| 65.2.005 | human | 3 | 0.712 | +0.916 | 0.0061 | 0.0057 | +0.332 | 0.31 |
| 70.1.008 | human | 0 | 0.862 | +0.958 | 0.0022 | 0.0015 | +0.104 | 0.35 |
| 70.1.008 | human | 1 | 0.843 | +0.963 | 0.0053 | 0.0060 | +0.128 | 0.30 |
| 70.1.008 | human | 2 | 0.857 | +0.974 | 0.0033 | 0.0025 | +0.080 | 0.33 |
| 70.1.008 | human | 3 | 0.857 | +0.966 | 0.0058 | 0.0050 | +0.052 | 0.33 |

20 candidates with an object found, of 20. Distributions (median [min, max]):

- `obj_fidelity`: 0.8569 [0.6257, 0.9707]
- `bg_fidelity`: 0.9189 [0.6766, 0.9741]
- `E_loc`: 0.0055 [0.0008, 0.0115]
- `E_area`: 0.0782 [0.0080, 0.4677]

Per-clip median (obj IoU, bg rho): 35.1.003/vehicle (0.905, +0.726); 84.0.006/vehicle (0.840, +0.971); 65.2.005/vehicle (0.942, +0.859); 65.2.005/human (0.677, +0.919); 70.1.008/human (0.857, +0.964)

The magenta overlay in `*_fidelity.png` is segment(generated): a mask on the wrong thing is an instrument error, not a generator finding. Review it first.
Heading class of each candidate is the HUMAN's label (no validated instrument): fill `heading_labels.csv`.
Tolerances are NOT proposed here by design: the human and the reviewer set them from this table (SKILL §5), then step 7 runs.
