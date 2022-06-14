import pandas as pd
import pathlib
import json
import ast



if __name__ == '__main__':
    direct = pathlib.Path.cwd() / 'data' / 'breeding' / 'final_data_breeding.csv'
    df = pd.read_csv(str(direct))
    store = pathlib.Path.cwd() / 'data' / 'breeding' / 'traits'
    all_data = []
    for i, row in df.iterrows():
        hashmap = []
        nft1 = row['nft1']
        nft2 = row['nft2']
        nft1 = ast.literal_eval(nft1)
        nft2 = ast.literal_eval(nft2)
        child_metadata = row['child_metadata']
        if pd.isna(child_metadata):
            child_metadata = {}
        else:
            child_metadata = ast.literal_eval(child_metadata)
        hashmap = [nft1, nft2, child_metadata]
        data = {f'{i}' : hashmap}
        all_data.append(data)

    with open(str(store / 'data.json'), 'w') as file:
        json.dump(all_data, file)