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
from requestData import *
# Get an API key pair for free here: https://dashboard.blockchainapi.com/
MY_API_KEY_ID = "VAG3eU6AZLbTjm9"
MY_API_SECRET_KEY = "MryAXR7kl3oYEPG"

BLOCKCHAIN_API_RESOURCE = TheBlockchainAPIResource(
    api_key_id=MY_API_KEY_ID,
    api_secret_key=MY_API_SECRET_KEY
)



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

    with open(str(data_directory / 'sample_data.pickle'), 'rb') as file:
        cm_data = pickle.load(file)

    degods_hashlist = cm_data['sample_cmid_to_hashlist']['degods']
    http_client = Client("https://broken-sparkling-frog.solana-mainnet.quiknode.pro/cd7cc881807e0e218614e5f86ea1f62b9aa1fc8e/")
    sample_degod = 'Byonooy5ijnzDFcnBePRjk3QFAjiuyTx6kD35H19AACx'
    one_transaction = http_client.get_confirmed_signature_for_address2(sample_degod)
    results = one_transaction['result']
    transactions = {}
    for result in results:
        signature = result['signature']
        transactions[signature] = http_client.get_confirmed_transaction(signature)
    test = '4fBgnn296SP1Es3Pvk4kuVjjBN7VSoTtdUx4W3GpBUUosSs8GqXE7LPL7p2hfrgpfoXTYVedoCa4fyWEoqemqvKX'
    pprint.pprint(transactions[test])