# file: Fox_Prestige_data.txt
"""

Results cross-checked with Fox 2008, p. 139:

                         SS   df        MS          F        p
--------------------------------------------------------------
income              1131.90    1   1131.90   28.35***   < .001
education           1067.98    1   1067.98   26.75***   < .001
type                 591.16    2    295.58    7.40**      .001
income x type        951.77    2    475.89   11.92***   < .001
education x type     238.40    2    119.20    2.99        .056

Residuals           3552.86   89     39.92
--------------------------------------------------------------
Total              28346.88   97

"""
from eelbrain import *


filepath = 'Fox_Prestige_data.txt'

# load data
ds = load.txt.tsv(filepath, delimiter=None)

# exclude cases with missing type
ds2 = ds[ds['type'] != 'NA']

# ANOVA
print(test.anova('prestige', '(income + education) * type', ds=ds2))
