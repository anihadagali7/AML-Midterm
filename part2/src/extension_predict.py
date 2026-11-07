"""Replay a frozen extension bundle on an outcome-format CSV."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from experiments import ROOT
from extension_followthrough import infer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",default="fast2_neighbors_half")
    parser.add_argument("--bundle",type=Path)
    args=parser.parse_args()
    if args.bundle is None:
        manifest=json.loads((ROOT/"submissions/ani_extensions_manifest.json").read_text())
        args.bundle=ROOT/manifest["bundle_path"]
    if not args.bundle.exists():
        raise FileNotFoundError("Rebuild the native bundle with extension_followthrough.py final first")
    data=pd.read_csv(args.input)
    assert data.ID.is_unique and set(data.phase).issubset({"I","II","III"})
    predictions=infer(joblib.load(args.bundle),data)
    assert args.model in predictions,f"Available models: {list(predictions)}"
    result=pd.DataFrame(dict(ID=data.ID,label=np.clip(predictions[args.model],.001,.999)))
    assert np.isfinite(result.label).all()
    result.to_csv(args.output,index=False)
    print("Wrote",len(result),"predictions to",args.output)


if __name__=="__main__":main()
