"""Convert the public Mendeley XLSX data file into the JSON format used by this package.

Requires: pandas + python-calamine. This script is supplied for portability; the
research workflow uses a locally supplied workbook; raw and prepared data are not
committed to this repository.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

EXCEL_EPOCH = pd.Timestamp('1899-12-30')

def scalar(v):
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        delta = v - EXCEL_EPOCH
        return delta.total_seconds() / 86400.0
    if isinstance(v, np.generic):
        return v.item()
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('xlsx')
    ap.add_argument('--out', default='../ijpe_local/data/tablet_data_extracted.json')
    args=ap.parse_args()
    try:
        book=pd.ExcelFile(args.xlsx, engine='calamine')
        engine='calamine'
    except ImportError:
        book=pd.ExcelFile(args.xlsx, engine='openpyxl')
        engine='openpyxl'
    out={}
    for sheet in book.sheet_names:
        df=pd.read_excel(book, sheet_name=sheet, engine=engine)
        header=[str(c) for c in df.columns]
        rows=[header]
        for row in df.itertuples(index=False, name=None):
            rows.append([scalar(v) for v in row])
        out[sheet]=rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(out,f,ensure_ascii=False)
    print(args.out)
if __name__=='__main__': main()
