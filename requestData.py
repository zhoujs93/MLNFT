import pathlib
import pprint
from theblockchainapi import TheBlockchainAPIResource, \
    SolanaNetwork, SolanaCandyMachineContractVersion, SearchMethod
import time, json, requests
import pickle, subprocess
from collections import defaultdict
from collections import deque
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

    # try to retrieve the hashlist of every nft
    me_data = {}
    cm_v2 = ['Taiyo Robotics', 'Mindfolk', 'Boryoku Dragonz', 'Degods', 'Famous Fox',
             'Portals', 'Lfinity Flames', 'Catalina', 'Degen Ape Academy',
             'Aurory', 'Galactic Geckos', 'Nyan Heroes', 'Stoned Ape Club',
             'Male HODL Whales']
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

    me_data['magiceden_project_data'] = data

    sample = {}
    for k, v in data.items():
        for name in cm_v2:
            if name in k:
                sample[name] = v

    listings = defaultdict(list)
    index = 0
    for k, v in sample.items():
        symbol = v['symbol']
        print(f'Moving to {symbol}')

        result = requests.get(url = f'https://api-mainnet.magiceden.dev/v2/collections/{symbol}/listings?offset={index}&limit=5')
        result = result.json()
        listings[symbol] += result

    me_data['test_listings_data'] = listings

    whales = listings['solana_hodl_whales']
    pprint.pprint(whales[0])
    hashmap_cmid = {}
    errors = set()
    for k, v in listings.items():
        for value in v:
            if len(hashmap_cmid.get(k, [])) == 0:
                try:
                    token_address = value['tokenMint']
                    result = BLOCKCHAIN_API_RESOURCE.get_candy_machine_id_from_nft(
                        mint_address=token_address,
                        network=SolanaNetwork.MAINNET_BETA
                    )
                    hashmap_cmid[k] = result
                except Exception as e:
                    print(f'Error for {k} : {e}')
                    errors.add(k)
            else:
                break
    # add test data
    me_data['sample_cm_id'] = hashmap_cmid
    cmid_to_hashlist = {}
    # get hashlist of CMID
    for key, val in hashmap_cmid.items():
        cm_id = val['candy_machine_id']
        is_v2 = val['candy_machine_contract_version'] == 'v2'
        hashlist = get_hashlist(cm_id, base_dir, is_v2 = is_v2)
        cmid_to_hashlist[key] = hashlist

    