def fixed_split(data, test_n=250, val_n=125):
    te = data.iloc[-test_n:]
    va = data.iloc[-(test_n+val_n):-test_n]
    tr = data.iloc[:-(test_n+val_n)]
    return tr, va, te