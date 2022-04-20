import pathlib
import pprint
from theblockchainapi import TheBlockchainAPIResource, \
    SolanaNetwork, SolanaCandyMachineContractVersion, SearchMethod
from solana.rpc.api import Client
import time, json, requests
import pickle, subprocess
from collections import defaultdict
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
# Get an API key pair for free here: https://dashboard.blockchainapi.com/
MY_API_KEY_ID = "VAG3eU6AZLbTjm9"
MY_API_SECRET_KEY = "MryAXR7kl3oYEPG"

BLOCKCHAIN_API_RESOURCE = TheBlockchainAPIResource(
    api_key_id=MY_API_KEY_ID,
    api_secret_key=MY_API_SECRET_KEY
)


def get_all_candy_machines():
    try:
        assert MY_API_KEY_ID is not None
        assert MY_API_SECRET_KEY is not None
    except AssertionError:
        raise Exception("Fill in your key ID pair!")

    current_time = int(time.time())
    result = BLOCKCHAIN_API_RESOURCE.list_all_candy_machines()
    print(result.keys())
    print(f"Last updated approx. {(current_time - result['last_updated']) // 60} minutes ago.")
    print(f"There are a total of {len(result['config_addresses_v1'])} V1 candy machines.")
    print(f"There are a total of {len(result['config_addresses_v2'])} V2 candy machines.")
    print(f"There are a total of {len(result['config_addresses_magic-eden-v1'])} Magic Eden candy machines.")
    return result

def get_nft_analytics(mint_addresses, start_time = -1, end_time = -1):
    print(f"Retrieved {len(mint_addresses)} mint addresses.")
    analytics = BLOCKCHAIN_API_RESOURCE.get_nft_marketplace_analytics(
        mint_addresses=mint_addresses,
        start_time=start_time,
        end_time=end_time
    )
    return analytics

def get_closed_CM(public_key):
    print(f"Transactions for {public_key}")
    transactions = BLOCKCHAIN_API_RESOURCE.get_wallet_transactions(
        public_key,
        network=SolanaNetwork.MAINNET_BETA
    )
    for transaction in transactions:
        print(f"Transaction Signature: {transaction}")
    return transactions

def get_hashlist(cm_id, base_dir, is_v2 = True):
    rpc = 'https://polished-cool-surf.solana-mainnet.quiknode.pro/66bc31e049b46dd293092114e2d8717b3378033f/'
    t = '180'
    if is_v2:
        cmd = ['metaboss', 'snapshot', 'mints', '-c', cm_id, '-r', rpc, '-t', t, '--output', './', '--v2']
    else:
        cmd = ['metaboss', 'snapshot', 'mints', '-c', cm_id, '-r', rpc, '-t', t, '--output', './']
    p = subprocess.run(cmd, cwd=str(base_dir / 'hashlist'))
    filename = f'{cm_id}_mint_accounts.json'
    dir_hlist = base_dir / 'hashlist'
    with open(str(dir_hlist / filename), 'r') as file:
        hlist = json.load(file)
    return hlist


if __name__ == '__main__':
    # load all candy machines
    load = False
    base_dir = pathlib.Path.cwd()
    data_directory = pathlib.Path.cwd() / 'data'
    if load:
        with open(str(data_directory / 'cm_mapping.pickle'), 'rb') as file:
            result = pickle.load(file)
    else:
        result = get_all_candy_machines()
        save = False
        if save:
            with open(str(data_directory / 'cm_mapping.pickle'), 'wb') as file:
                pickle.dump(result, file)
    #
    # # try to retrieve the hashlist of every nft
    # me_data = {}
    # cm_v2 = ['Taiyo Robotics', 'Mindfolk', 'Boryoku Dragonz', 'Degods',
    #          'Portals', 'Catalina', 'Degenerate Ape Academy',
    #          'Aurory', 'Galactic Geckos', 'Nyan Heroes', 'Stoned Ape Crew',
    #          'Male HODL Whales', 'Cets On Creck', 'Solana Monkey Business', 'Zaysan Raptors',
    #          'Degenerate Trash Pandas', 'Dazed']
    # results = []
    # index = 0
    # try:
    #     while True:
    #         result = requests.get(url = f'https://api-mainnet.magiceden.dev/v2/collections?offset={index}&limit=500')
    #         result = result.json()
    #         pprint.pprint(result[-1])
    #         index += 500
    #         results += result
    #         print(f'len of results is {len(results)}')
    # except Exception as e:
    #     print(f'{index} : {e}')
    #
    # data = {}
    # for collection in results:
    #     name = collection['name']
    #     data[name] = collection
    #
    # me_data['magiceden_project_data'] = data
    #
    # sample = {}
    # for k, v in data.items():
    #     proc_name = k.replace(' ', '').lower()
    #     for name in cm_v2:
    #         proc_key = name.replace(' ', '').lower()
    #         if proc_key in proc_name:
    #             sample[k] = v

    #
    # listings = defaultdict(list)
    # index = 0
    # for k, v in sample.items():
    #     symbol = v['symbol']
    #     print(f'Moving to {symbol}')
    #
    #     result = requests.get(url = f'https://api-mainnet.magiceden.dev/v2/collections/{symbol}/listings?offset={index}&limit=5')
    #     result = result.json()
    #     listings[symbol] += result
    #
    # me_data['test_listings_data'] = listings
    #
    # whales = listings['solana_hodl_whales']
    #
    # hashmap_cmid = {}
    # errors = set()
    # for k, v in listings.items():
    #     for value in v:
    #         if len(hashmap_cmid.get(k, [])) == 0:
    #             try:
    #                 token_address = value['tokenMint']
    #                 result = BLOCKCHAIN_API_RESOURCE.get_candy_machine_id_from_nft(
    #                     mint_address=token_address,
    #                     network=SolanaNetwork.MAINNET_BETA
    #                 )
    #                 hashmap_cmid[k] = result
    #             except Exception as e:
    #                 print(f'Error for {k} : {e}')
    #                 errors.add(k)
    #         else:
    #             break
    # # add test data
    # me_data['sample_cm_id'] = hashmap_cmid
    cmid_to_hashlist = {}
    # get hashlist of CMID
    # for key, val in hashmap_cmid.items():
    #     cm_id = val['candy_machine_id']
    #     is_v2 = val['candy_machine_contract_version'] == 'v2'
    # cmid = 'DGPM2TfpoCYsBoTjsAttbCf6HWuGwqPBpPn3jyyXDwYW'
    # is_v2 = True
    # hashlist = get_hashlist(cmid, base_dir, is_v2 = is_v2)
    with open('./data/baby_hashlist.json', 'r') as file:
        hashlist = json.load(file)

    data = []
    hlist = []
    for i, nft in enumerate(hashlist):
        print(i)
        hlist.append(nft)
        nft_metadata = BLOCKCHAIN_API_RESOURCE.get_nft_metadata(
            mint_address= nft,
            network=SolanaNetwork.MAINNET_BETA
        )
        nft_data = {
            'mint_id' : nft_metadata['mint'],
            'metadata' : nft_metadata['off_chain_data']
        }
        data.append(nft_data)

    with open('./data/metadata.json', 'w') as file:
        json.dump(data, file)


    #
    # me_data['sample_cmid_to_hashlist'] = cmid_to_hashlist
    # transaction_errors = defaultdict(list)
    # transaction_data = {}
    # for k, v in cmid_to_hashlist.items():
    #     print(f'Processing {k}')
    #     temp_tx = {}
    #     for i in range(0, len(v), 250):
    #         try:
    #             hlist = v[i : i + 250]
    #             tx_data = get_nft_analytics(hlist, -1, None)
    #             temp_tx.update(tx_data['transaction_history'])
    #         except Exception as e:
    #             print(f'Error for {k} on ({i}, {i+250}) : {e}')
    #             transaction_errors[k] += hlist
    #     transaction_data[k] = temp_tx
    #
    # me_data['sample_transaction_data'] = transaction_data
    # with open(str(data_directory / 'sample_data.pickle'), 'wb') as file:
    #     pickle.dump(me_data, file)
    #
    # transaction_data = me_data['sample_transaction_data']
    # all_dfs = {}
    # for name, val in transaction_data.items():
    #     project_df = {}
    #     for k, v in val.items():
    #         if len(v) != 0:
    #             project_df[k] = pd.DataFrame(v)
    #     project_df = pd.concat(list(project_df.values()), axis = 0, ignore_index = True)
    #     all_dfs[name] = project_df
    #
    # for k, v in all_dfs.items():
    #     v['project_name'] = k
    #     all_dfs[k] = v
    # dfs = pd.concat(list(all_dfs.values()), axis = 0, ignore_index = True)
    # dfs['datetime'] = pd.to_datetime(dfs['block_time'], unit = 's')
    # sample = dfs.loc[dfs['project_name'] == 'degods', :]
    # buy_df = sample.loc[sample['operation'] == 'buy']
    # buy_df['price'] = buy_df['price'] / 10**9
    # avg_price = buy_df.resample('D', on = 'datetime').mean()
    # ax = avg_price['price'].plot(y = 'price', figsize = (12, 12), title = 'average price per day')
    # plt.show()
    #
    # http_client = Client("https://broken-sparkling-frog.solana-mainnet.quiknode.pro/cd7cc881807e0e218614e5f86ea1f62b9aa1fc8e/")
    # one_transaction_set = http_client.get_confirmed_signature_for_address2('6RiLfUjMZsysLWCM5r2uhnrzFyQmTev1aHcJLBTVy7ss')
    # one_transaction = http_client.get_confirmed_transaction('4kgeHWWTRKtJyfRNETtzaqfgmaDVYLkXmiyCWtTWQAnYXsauPoVEghfGeSmBSai6rJdAkZjWQafJctp7Amwk3qmQ')
    # # one_transaction_tr = http_client.get_confirmed_signature_for_address2('MEisE1HzehtrDpAAT8PnLHjpSSkRYakotTuJRPjTpo8')
    # tx_signature = []
    # me_transactions = {}
    # index = 0
    # min_date = pd.to_datetime('2021-12-01')
    # cur_min_date = pd.to_datetime('2021-12-30')
    # try:
    #     last_tx = 'kxsDBjRtvudXUPmXcAoUnG8WsgQv5Z89Q2n6kQo5cu9ksPs14S21JUyiotyLLiBPNjvnncRPCkz1jcSQLKQHCvB'
    #     while True:
    #         if index % 10 == 0:
    #             print(f'Index = {1000 * index}')
    #         program_acc = http_client.get_confirmed_signature_for_address2(
    #             account = 'MEisE1HzehtrDpAAT8PnLHjpSSkRYakotTuJRPjTpo8',
    #             before= last_tx
    #         )
    #         last_tx = program_acc['result'][-1]['signature']
    #         for result in program_acc['result']:
    #             tx_signature.append(result)
    #             tx_sig = result['signature']
    #             me_transactions[tx_sig] = result
    #             blocktime = pd.to_datetime(result['blockTime'], unit = 's')
    #             cur_min_date = min(cur_min_date, blocktime)
    #         if cur_min_date <= min_date:
    #             break
    #         index += 1
    # except Exception as e:
    #     print(f'Exception received: {e}')
    #
    #
    #
    #
