import pathlib
import pprint
from theblockchainapi import TheBlockchainAPIResource, \
    SolanaNetwork, SolanaCandyMachineContractVersion, SearchMethod
import time, json, requests
import pickle, subprocess
from collections import defaultdict
from collections import deque
# Get an API key pair for free here: https://dashboard.blockchainapi.com/
MY_API_KEY_ID = ""
MY_API_SECRET_KEY = ""

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

    start_time = None  # Default start time is 1 day ago. Provide -1 to get full history (since we began recording it).
    end_time = None
    print(f"Retrieved {len(mint_addresses)} mint addresses.")
    analytics = BLOCKCHAIN_API_RESOURCE.get_nft_marketplace_analytics(
        mint_addresses=mint_addresses,
        start_time=start_time,
        end_time=end_time
    )
    print(f"NFT Transactions: {json.dumps(analytics, indent=4)}")
    print(f"Floor = {analytics['floor']}")
    print(f"Volume = {analytics['volume']}")
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

    # try to retrieve the hashlist of every nft
    cm_v2 = ['Taiyo Robotics', 'Mindfolk', 'Boryoku Dragonz', 'Degods', 'Famous Fox',
             'Portals', 'Lfinity Flames', 'Catalina', 'Degen Ape Academy',
             'Aurory', 'Galactic Geckos', 'Nyan Heroes', 'Stoned Ape Club']
    results = []
    index = 0
    try:
        while True:
            result = requests.get(url = f'https://api-mainnet.magiceden.dev/v2/collections?offset={index}&limit=500')
            result = result.json()
            pprint.pprint(result[-1])
            index += 500
            results += result
            print(f'len of results is {len(results)}')
    except Exception as e:
        print(f'{index} : {e}')

    data = {}
    for collection in results:
        name = collection['name']
        data[name] = collection

    sample = {}
    for k, v in data.items():
        for name in cm_v2:
            if name in k:
                sample[name] = v

    tx_data = defaultdict(list)
    for k, v in sample.items():
        symbol = v['symbol']
        print(f'Moving to {symbol}')
        try:
            index = 0
            results = []
            while True:
                if index % 100 == 0:
                    print(f'{symbol} : {index}')
                time.sleep(5)
                result = requests.get(url = f'https://api-mainnet.magiceden.dev/v2/collections/{symbol}/'
                                            f'listings?offset={index}&limit=100')
                result = result.json()
                tx_data[symbol] += result
                index += 100
        except Exception as e:
            print(f'{index} : {e}')

