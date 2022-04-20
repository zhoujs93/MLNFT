import time
from bs4 import BeautifulSoup
from selenium import webdriver
import pathlib, pickle
import json, pprint
import ray
import pandas as pd
import requests
from collections import deque, defaultdict
ray.init()


@ray.remote
def scrape_project(k, v):
    print(f'Moving on to {k}')
    timeout = 40
    errors = []
    hashmap = {}
    # options = webdriver.ChromeOptions()
    # options.add_argument('window-size=1920x1080')
    for i, sample in enumerate(v):
        if i % 10 == 0:
            print(f'Moving onto {i}')
        try:
            url = f'https://api-mainnet.magiceden.dev/rpc/getGlobalActivitiesByQuery?q=%7B%22$match%22:%7B%22mint%22:%22{sample}%22%7D,%22$sort%22:%7B%22blockTime%22:-1,%22createdAt%22:-1%7D,%22$skip%22:0%7D'
            data = requests.get(url)
            data = data.json()
            hashmap[sample] = data
            # driver = webdriver.Chrome(executable_path='./chromedriver')
            # driver.minimize_window()
            # driver.get(url)
            # driver.implicitly_wait(10)
            # soup_level = BeautifulSoup(driver.page_source, 'lxml')
            # data = soup_level.text
            # tx = json.loads(data)
            # print(f'Length of data is {len(tx.get("results", []))} for {sample}')
            # hashmap[sample] = tx
            # driver.close()
        except Exception as e:
            print(f'Error Occurred for {sample} : {e}')
            errors.append(sample)
    data_map = {
        'transactions': hashmap,
        'errors': errors
    }
    with open(str(data_dir / f'{k}.pickle'), 'wb') as file:
        pickle.dump(data_map, file)
    return data_map

@ray.remote
def scrape_url(sample, index):
    if index % 10 == 0:
        print(f'Moving onto {index}')
    data, request = None, None
    while data is None:
        try:
            url = f'https://api-mainnet.magiceden.dev/rpc/getGlobalActivitiesByQuery?q=%7B%22$match%22:%7B%22mint%22:%22{sample}%22%7D,%22$sort%22:%7B%22blockTime%22:-1,%22createdAt%22:-1%7D,%22$skip%22:0%7D'
            request = requests.get(url)
            data = request.json()
        except Exception as e:
            print(f'Error Occurred for {sample} : {e}')
            data = None
            time.sleep(60)
    return (sample, data, request)

def convert_hash_to_df(project, hashmap):
    dfs_sales = []
    for k, v in hashmap.items():
        df = pd.DataFrame(v)
        df['mint_id'] = k
        df['project'] = project
        dfs_sales.append(df)
    sales_df = pd.concat(dfs_sales, axis=0)
    return sales_df



if __name__ == '__main__':
    url = 'https://magiceden.io/item-details/'
    base_dir = pathlib.Path.cwd()
    data_dir = pathlib.Path.cwd() / 'data'

    with open(str(data_dir / 'sample_data.pickle'), 'rb') as file:
        cm_data = pickle.load(file)

    # mappings = ['aurory', 'cets_on_creck', 'ggsg:_galactic_geckos', 'catalina_whale_mixer',
    #             'taiyo_robotics', 'female_hodl_whales', 'degods', 'stoned_ape_crew']
    mappings = ['solana_hodl_whales']

    howrare = pd.read_pickle(str(data_dir / 'howrare_df.pickle'))
    howrare['project_name'] = howrare['link'].apply(lambda x: x.split('/')[-2])
    projects = howrare['project_name'].unique()
    sample = '6CCprsgJT4nxBMSitGathXcLshDTL3BE4LcJXvSFwoe2'
    url = f'https://api-mainnet.magiceden.dev/rpc/getGlobalActivitiesByQuery?q=%7B%22$match%22:%7B%22mint%22:%22{sample}%22%7D,%22$sort%22:%7B%22blockTime%22:-1,%22createdAt%22:-1%7D,%22$skip%22:0%7D'
    request = requests.get(url)
    hashmap, i = {}, 0
    for project in projects:
        print(f'Moving onto {project}')
        if project != 'degods':
            results = []
            mint_ids = howrare.loc[howrare['project_name'] == project, 'mint'].unique()
            for i, sample in enumerate(mint_ids):
                results.append(scrape_url.remote(sample, i))
            result = ray.get(results)
            hashmap[project] = result

    all_sales, all_listings, all_unlistings, all_othertx = {}, {}, {}, {}
    for project, value in hashmap.items():
        sales, listings, unlistings, othertx = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for item in value:
            mint_id, data, response = item
            if len(data.get('results', [])) != 0:
                for tx in data['results']:
                    if 'parsedUnlist' in tx:
                        tx['parsedUnlist']['blockTime'] = tx['blockTime']
                        unlistings[mint_id].append(tx['parsedUnlist'])
                    elif 'parsedList' in tx:
                        tx['parsedList']['blockTime'] = tx['blockTime']
                        listings[mint_id].append(tx['parsedList'])
                    elif 'parsedTransaction' in tx:
                        sales[mint_id].append(tx['parsedTransaction'])
                    else:
                        othertx[mint_id].append(tx)
        dfs_sales = []
        sales_df = convert_hash_to_df(project, sales)
        listings_df = convert_hash_to_df(project, listings)
        delists_df = convert_hash_to_df(project, unlistings)
        othertx_df = convert_hash_to_df(project, othertx)
        all_sales[project] = sales_df
        all_listings[project] = listings_df
        all_unlistings[project] = delists_df
        all_othertx[project] = othertx_df

    df_sales = pd.concat(list(all_sales.values()), axis = 0)
    df_listings = pd.concat(list(all_listings.values()), axis = 0)
    df_delists = pd.concat(list(all_unlistings.values()), axis = 0)
    df_other = pd.concat(list(all_othertx.values()), axis = 0)

    df_sales = df_sales.merge(howrare, how = 'left', on = 'mint')
    df_listings = df_listings.merge(howrare, how = 'left', on = 'mint')
    df_delists = df_delists.merge(howrare, how = 'left', on = 'mint')
    df_other = df_other.merge(howrare, how = 'left', on = 'mint')
    all_dfs = {
        'sales' : df_sales,
        'listings' : df_listings,
        'delists' : df_delists,
        'other' : df_other
    }
    with open(str(data_dir / 'hodl_whales_data.pickle'), 'wb') as file:
        pickle.dump(all_dfs, file)

    #
    # with open(str(data_dir / 'degods_final.pickle'), 'wb') as file:
    #     pickle.dump(degods_data, file)
    # #
    # listings have 'parsedList'
    # unlistings have 'parsedUnlist'
    # sales have 'parsedTransactions'
    #
    #     print(f'Moving on to {project}')
    #     mint_df = howrare.loc[howrare['project_name'] == project]
    #     hashmap = {}
    #     v = mint_df['mint'].unique()
    #     result.append(scrape_project.remote(project, v))
    # result = ray.get(result)
    # # This function should predict classes for new items in the testing data