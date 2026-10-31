"""Chronological data preparation."""
import pandas as pd

def chronological_split(frame, fractions=(.70,.15,.15)):
    if abs(sum(fractions)-1)>1e-8: raise ValueError('fractions must sum to one')
    n=len(frame); a=int(n*fractions[0]); b=a+int(n*fractions[1])
    return frame.iloc[:a],frame.iloc[a:b],frame.iloc[b:]
