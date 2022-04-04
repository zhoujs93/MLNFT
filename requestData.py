import pathlib
from theblockchainapi import TheBlockchainAPIResource, SolanaNetwork, SolanaCandyMachineContractVersion
import time, json, requests
import pickle, subprocess
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
    # public_key = '2VxSwAFYDx82gbKZTXsa5dieg3V9ZxbcZtnrUuv8WJ6g'
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
    cm_v2 = result['config_addresses_v2']
    cm_to_hashlist = {}
    errors = {}
    cm_v2_test = cm_v2[:100]
    # for cmid in cm_v2_test:
    #     try:
    #         result = BLOCKCHAIN_API_RESOURCE.get_all_nfts_from_candy_machine(
    #             candy_machine_id=cmid,
    #             network=SolanaNetwork.MAINNET_BETA
    #         )
    #         cm_to_hashlist[cmid] = result
    #     except Exception as e:
    #         print(f'{e}')
    #         errors[cmid] = e



    # hodl_whales in result['config_addresses_v2']
    # all_nfts = get_cm_nft(hodl_whales, verbose = False, v2 = True)
    # all_nft_info, minted_nfts, unminted_nfts = all_nfts['all_nfts'], all_nfts['minted_nfts'], all_nfts['unminted_nfts']
    # addresses = []
    # for metadata in minted_nfts:
    #     mint_account = metadata['nft_metadata']['mint']
    #     addresses.append(mint_account)

    # analytics = get_nft_analytics(addresses)
    # test = 'F7UWhEUnvbD67Lj5DrYyMvtRFC5XETHeJvRycydDJsu7'
    # test_nft_tx = analytics['transaction_history'][test]
    # data = {k: v for k, v in analytics['transaction_history'].items() if len(v) != 0}
    # data_dir = pathlib.Path.cwd() / 'data' / 'HODL_data.json'
    # with open(str(data_dir), 'w') as file:
    #     json.dump(data, file)


    # get unsigned tx
    # hodl_whales = '2VxSwAFYDx82gbKZTXsa5dieg3V9ZxbcZtnrUuv8WJ6g'
    # transactions = get_closed_CM(hodl_whales)
    # tx_details = []
    # for transaction in transactions[:5]:
    #     transaction = BLOCKCHAIN_API_RESOURCE.get_solana_transaction(
    #         tx_signature=transaction,
    #         network=SolanaNetwork.MAINNET_BETA
    #     )
    #     print(transaction)
    #     tx_details.append(transaction)
    # direct = pathlib.Path.cwd() / 'data' / 'test_tx.json'
    # with open(direct, 'w') as file:
    #     json.dump(tx_details, file)
