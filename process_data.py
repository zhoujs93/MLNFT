import pandas as pd
import pickle, pprint, pathlib, json
import requests
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def get_howrare():
    url = 'https://api.howrare.is/v0.1/collections'
    request = requests.get(url)
    request = request.json()
    return request

def flatten_data(data):
    for i in range(len(data)):
        item = data[i]
        key = None
        for k, v in item.items():
            if type(v) is dict:
                key = k
                break
        if key is not None:
            item.update(item[key])
            del item[key]
        data[i] = item
    return data

def process_how_rare_data():
    howrare = get_howrare()
    howrare_data = howrare['result']['data']
    howrare_df = pd.DataFrame(howrare_data)
    howrare_df['name'] = howrare_df['name'].apply(lambda x: x.lower().replace(' ', '_'))
    return howrare_df

def get_collection_data(data_dir, fname, collection_name):
    with open(str(data_dir / fname), 'rb') as file:
        tx_data = pickle.load(file)
    dfs = {}
    result = tx_data['transactions']
    for hash_id, value in result.items():
        results = []
        for res in value.get('results', []):
            if 'parsedTransaction' in res:
                results.append(res['parsedTransaction'])
        if len(results) != 0:
            temp = pd.DataFrame(results)
            dfs[hash_id] = temp
    df_all = pd.concat(list(dfs.values()), axis = 0)
    df_all['datetime'] = pd.to_datetime(df_all['blockTime'], unit = 's')
    df_all = df_all.assign(total_amount = lambda x: x['total_amount'] / 10**9)
    df_all['collections'] = collection_name
    return df_all

def get_howrare_collection(url):
    base = f'https://api.howrare.is/v0.1/collections/{url}/only_rarity'
    collection_rarity = requests.get(base)
    rarity = collection_rarity.json()
    rarity_data = rarity['result']['data']['items']
    rarity_data = flatten_data(rarity_data)
    rarity_df = pd.DataFrame(rarity_data)
    return rarity_df


if __name__ == '__main__':
    data_dir = pathlib.Path.cwd() / 'data'
    files = {
        'aurory' : 'aurory.pickle', 'cets_on_creck' : 'cets_on_creck.pickle',
        'degods' : 'degods.pickle', 'galactic_geckos' : 'galactic_geckos.pickle',
        'female_hodl_whales' : 'female_hodl_whales.pickle', 'taiyo_robotics' : 'taiyo_robotics.pickle',
        'stoned_ape_crew' : 'stoned_ape_crew.pickle', 'the_catalina_whale_mixer' : 'the_catalina_whale_mixer.pickle'
    }
    dfs = []
    for k, v in files.items():
        df = get_collection_data(data_dir, v, k)
        dfs.append(df)

    df_all = pd.concat(dfs, axis = 0, ignore_index = True)

    mappings = ['aurory', 'cets_on_creck', 'ggsg:_galactic_geckos', 'catalina_whale_mixer',
                'taiyo_robotics', 'female_hodl_whales', 'degods', 'stoned_ape_crew']

    howrare_df = pd.read_excel(str(data_dir / 'howrare_data.xlsx'))
    howrare_data = howrare_df.loc[howrare_df['name'].isin(mappings)]
    howrare_data['url'] = howrare_data['url'].apply(lambda x: x.replace('/', ''))
    urls = howrare_data['url'].tolist()
    howrare_dfs = []
    for url in urls:
        rarity_df = get_howrare_collection(url)
        howrare_dfs.append(rarity_df)
    howrare_df = pd.concat(howrare_dfs, axis = 0)

    df_merged = df_all.merge(howrare_df, how = 'left', left_on = 'mint', right_on = 'mint')
    df_merged = df_merged.sort_values(by = 'datetime', ignore_index = True)
    df_merged['last_price'] = df_merged.groupby(['mint'])['total_amount'].apply(lambda x: x.shift(1))
    df_merged['price_change'] = df_merged.groupby(['mint'])['total_amount'].apply(lambda x: x.pct_change(1) - 1)
    df_merged['target'] = df_merged.groupby(['mint'])['price_change'].apply(lambda x: x.shift(-1))

    features = ['collection_symbol', 'mint', 'total_amount', 'seller_address', 'buyer_address',
                'datetime', 'rank', 'howrare.is', 'trait_normalized', 'statistical_rarity',
                'price_change', 'last_price', 'target']
    df = df_merged[features]
    df = df.assign(year = lambda x: x['datetime'].dt.year, month = lambda x: x['datetime'].dt.month,
                   week = lambda x: x['datetime'].dt.week)
    df_filtered = df.copy()
    # TODO: explore other options
    # df_filtered[['last_price', 'price_change']] = df_filtered[['last_price', 'price_change']].fillna(0.0)
    df_filtered['total_weekly_volume'] = df_filtered.groupby(['week', 'mint'])['last_price'].transform(lambda x: x.sum())
    df_filtered['total_monthly_volume'] = df_filtered.groupby(['month', 'mint'])['last_price'].transform(lambda x: x.sum())
    # filter outliers
    change = 0.85
    # get daily calculations
    projects = df_filtered['collection_symbol'].unique()
    ts_dir = pathlib.Path.cwd() / 'data' / 'ts-data'
    for test in projects:
        print(test)
        try:
            sample = df_filtered.loc[df_filtered['collection_symbol'] == test]
            sample = sample.loc[(sample['price_change'].abs() <= change), :].reset_index(drop = True)
            daily_df = sample.resample('D', on = 'datetime').apply(lambda x: x['total_amount'].mean())
            med_df = (sample.resample('D', on = 'datetime').apply(lambda x: x['total_amount'].median()))
            floor_df = (sample.resample('D', on = 'datetime').apply(lambda x: x['total_amount'].min()))
            volume_df = (sample.resample('D', on = 'datetime').apply(lambda x: x['total_amount'].sum()))
            count_df = (sample.resample('D', on = 'datetime').apply(lambda x: x.count()))
            daily_df = pd.concat([daily_df, med_df, floor_df], axis = 1).fillna(method = 'ffill')
            daily_df.columns = ['Avg Price', 'Median Price', 'Floor Price']
            ax = daily_df.plot(y = ['Avg Price', 'Median Price', 'Floor Price'],
                               figsize = (12, 12), title = f'{test} Price Plots')
            plot_dir = pathlib.Path.cwd() / 'plots'
            plt.savefig(str(plot_dir / f'{test}.jpeg'))
            plt.show()
            daily_df.to_csv(str(ts_dir / f'{test}.csv'))

        except Exception as e:
            print(e)


    # min_dates = df_all.resample('D', on = 'datetime').apply(lambda x: x['total_amount'].min())
    # min_dates = min_dates.reset_index(drop = False).rename({0 : 'floor_price'}, axis = 1)
    # min_dates['pct_change'] = min_dates['floor_price'].pct_change(1)
    # outlier_pct = 0.85
    # idx = (min_dates['pct_change'].abs() <= outlier_pct)
    # min_dates = min_dates.loc[(min_dates['pct_change'].abs() <= outlier_pct), :].reset_index(drop = True)
    # min_dates['floor_price'] = min_dates['floor_price'].fillna(method = 'backfill')
    # ax = min_dates.plot(x = 'datetime', y = 'floor_price', figsize = (12, 12))
    # plt.show()

