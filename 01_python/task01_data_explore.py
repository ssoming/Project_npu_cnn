"""
WM-811K pkl 파일 데이터 확인 스크립트
"""

import pickle
import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes


PKL_PATH = '../data/LSWMD.pkl'
OUT_PATH = '../data/wafer_data.csv'


# 1. pkl 로드
print("[1] pkl 로드 중...")
with open(PKL_PATH, 'rb') as f:
    df = pickle.load(f, encoding='latin1')
print('pkl column',df.columns.tolist())

# 중첩 리스트 [['값']] → '값' 언패킹 함수
def unpack(val):
    try:
        return val[0][0]
    except:
        return ''

# 2. waferMap, trainTestLabel 제외, failureType null 제거 후 저장
print("[2] CSV 저장 중...")
result_df = pd.DataFrame({
    'dieSize'     : df['dieSize'].astype(int),
    'lotName'     : df['lotName'],
    'waferIndex'  : df['waferIndex'].astype(int),
    'failureType' : df['failureType'].apply(unpack),
})
result_df = result_df[result_df['failureType'] != '']
result_df.to_csv(OUT_PATH, index=False, encoding='utf-8-sig')
print(f"  → 저장 완료: {OUT_PATH}  ({len(result_df):,} rows)")