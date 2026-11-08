"""Use the retained-evidence verifier with the v2 model configuration."""
import sys
from pathlib import Path
import strong_study as wrapper
import retained_evidence

if __name__=="__main__":
    command=sys.argv[1] if len(sys.argv)>1 else "verify"
    directory=Path(sys.argv[2]) if len(sys.argv)>2 else wrapper.HERE/"results"
    if command=="compact":retained_evidence.compact(directory,
        kept=['parent','anchored','calibration','head_ordinary','head_diverse30',
              'blend_ordinary','blend_diverse30','fusion_ordinary','fusion_diverse30'],
        rationale='Retain the matched lambda0/.3 heads, fixed blends and learned fusion, plus calibration and established controls. This preserves the small fixed-blend diversity benefit alongside its standalone accuracy cost. Intermediate .1 predictions and all fold caches were independently verified before removal; every summary is kept.')
    else:retained_evidence.verify(directory)
