from collections import Counter, defaultdict
from solana.rpc.api import Client
from pprint import pprint
import requests
import time, random
from theblockchainapi import TheBlockchainAPIResource, SolanaCurrencyUnit, SolanaNetwork, SolanaWallet, DerivationPath
import pathlib
import json, pickle, base58
import pandas as pd
from collections import defaultdict, deque

# Get an API key pair for free here: https://dashboard.blockchainapi.com/
MY_API_KEY_ID = "VAG3eU6AZLbTjm9"
MY_API_SECRET_KEY = "MryAXR7kl3oYEPG"

BLOCKCHAIN_API_RESOURCE = TheBlockchainAPIResource(
    api_key_id=MY_API_KEY_ID,
    api_secret_key=MY_API_SECRET_KEY
)


if __name__ == '__main__':
    directory = pathlib.Path.cwd() / 'data' / 'breeding'
    with open(str(directory / 'all_data_tx.json'), 'r') as file:
        all_data = json.load(file)

    transferred = Counter()
    for data in all_data:
        if 'mint_id' in data and data['transferred'] == True:
            transferred[(data['wallet'], data['mint_id'])] += 1

    total = sum(transferred.values())
    with open(str(directory / 'gib-holders.json'), 'r') as file:
        holders = json.load(file)

    mint_wallet = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
    mints = holders[mint_wallet]['mints']
    all_data_cleaned = []
    for data in all_data:
        nft1 = data['nft1']['metadata']['name']
        mintid1 = data['nft1']['mint_id']
        data['nft1_name'] = nft1
        data['nft1_mint_id'] = mintid1
        nft2 = data['nft2']['metadata']['name']
        mintid2 = data['nft2']['mint_id']
        data['nft2_name'] = nft2
        data['nft2_mint_id'] = mintid2
        all_data_cleaned.append(data)

    df = pd.DataFrame(all_data_cleaned)
    unique_mint_id = len(df['mint_id'].unique())
    df_na = df.loc[df['mint_id'].isna()]
    total_na = df_na.shape[0]
    test_mint = 'drRz4e8cA9k9okHA1yvF9n81GC3jiKYMbEG6nL1zbPg'
    test_df = df.loc[df['mint_id'] == test_mint]
    dup_wallet = '4zVQGXh4AvkHtpoh9CAGFzwPgvqqSUFTyn3GDT7hgHgs'
    # df.to_excel('./data/breeding/transactions.xlsx', index = False)
    all_mints = []
    wallets = []
    for k, v in holders.items():
        if k != mint_wallet:
            all_mints += v['mints']
        wallets.append(k)

    df_dup = df.loc[(df['mint_id'].isin(all_mints) & (~df['wallet'].isin(wallets)))]
    df_parents = df.groupby(['nft1_name', 'nft2_name']).size().reset_index().rename({0 : 'count'}, axis = 1)
    bred_multiple = df_parents.loc[df_parents['count'] > 1, :]
    bred_mulitple_nft1 = bred_multiple['nft1_name'].unique().tolist()
    bred_multiple_nft2 = bred_multiple['nft2_name'].unique().tolist()
    df_bred = df.loc[(df['nft1_name'].isin(bred_mulitple_nft1)) & (df['nft2_name'].isin(bred_multiple_nft2))]
    max_nft1 = 'Solana HODL Whales #245'
    max_nft2 = 'Solana Female HODL Whales #152'
    df_max = df_bred.loc[(df_bred['nft1_name'] == max_nft1) & (df_bred['nft2_name'] == max_nft2)]
    # bred_multiple.to_csv('./data/breeding/bred_multiple_times.csv')
    bred_multiple['actual_count'] = bred_multiple['count'] - 1
    df_max.to_excel('./data/breeding/outlier.xlsx')

    df_parents[['nft1_name','nft2_name']].to_excel('./data/breeding/bred_tools.xlsx')

    http_client = Client("https://polished-cool-surf.solana-mainnet.quiknode.pro/66bc31e049b46dd293092114e2d8717b3378033f/")
    breed_token_account = '6oW51vsAy4gHmapEjyYWmCoVSdAYxSuAysVxB4LEc3Pa'
    account_tx = http_client.get_signatures_for_address(breed_token_account)
    signatures = {}
    for account in account_tx['result']:
        sig = account['signature']
        signatures[sig] = account


    all_tx_pre = defaultdict(list)
    all_tx_post = defaultdict(list)
    errors = {}
    for i, (signature, account) in enumerate(signatures.items()):
        try:
            details = http_client.get_confirmed_transaction(signature)
            meta = details['result']['meta']
            post_bal = meta['postTokenBalances']
            pre_bal = meta['preTokenBalances']
            wallet = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
            pre_bal_recipient = 0
            pre_bal_sender = 0
            first_wallet_add, first_wallet_amount = pre_bal[0]['owner'], float(pre_bal[0]['uiTokenAmount']['amount']) / 10 ** 9
            second_wallet_add, second_wallet_amount = pre_bal[1]['owner'], float(pre_bal[1]['uiTokenAmount']['amount']) / 10**9

            first_wallet_add_post, first_wallet_amount_post = post_bal[0]['owner'], float(post_bal[0]['uiTokenAmount']['amount']) / 10 ** 9
            second_wallet_add_post, second_wallet_amount_post = post_bal[1]['owner'], float(post_bal[1]['uiTokenAmount']['amount']) / 10 ** 9
            all_tx_pre['date'].append(account['blockTime'])
            all_tx_pre['transaction'].append(signature)
            all_tx_pre['first_wallet'].append(first_wallet_add)
            all_tx_pre['first_wallet_pre_bal'].append(first_wallet_amount)
            all_tx_pre['second_wallet'].append(second_wallet_add)
            all_tx_pre['second_wallet_pre_bal'].append(second_wallet_amount)
            all_tx_post['transaction'].append(signature)
            all_tx_post['first_wallet'].append(first_wallet_add_post)
            all_tx_post['first_wallet_post_bal'].append(first_wallet_amount_post)
            all_tx_post['second_wallet'].append(second_wallet_add_post)
            all_tx_post['second_wallet_post_bal'].append(second_wallet_amount_post)
        except Exception as e:
            print(f'{e}')
            errors[signature] = details

    df_post = pd.DataFrame(all_tx_post)
    # df_post = df_post.rename({'first_wallet_pre_bal' : 'first_wallet_post_bal',
    #                           'second_wallet_pre_bal' : 'second_wallet_post_bal'}, axis = 1)
    df_pre = pd.DataFrame(all_tx_pre)
    df_krill = df_post.merge(df_pre, how = 'left', left_on = ['transaction', 'first_wallet', 'second_wallet'],
                             right_on = ['transaction', 'first_wallet', 'second_wallet'])
    df_krill = df_krill.assign(first_wallet_diff = lambda x: x['first_wallet_post_bal'] - x['first_wallet_pre_bal'],
                               second_wallet_diff = lambda x: x['second_wallet_post_bal'] - x['second_wallet_pre_bal'])
    df_krill['date'] = pd.to_datetime(df_krill['date'], unit = 's')
    df_krill['recipient'] = df_krill['first_wallet']
    breed_wallet = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
    index = (df_krill['first_wallet_diff'] < 0) & (df_krill['second_wallet_diff'] > 0) & (df_krill['second_wallet'] == breed_wallet)
    df_krill.loc[index, 'recipient'] = df_krill.loc[index, 'second_wallet']
    df_krill['sender'] = df_krill['first_wallet']
    df_krill.loc[~index, 'sender'] = df_krill.loc[~index, 'second_wallet']
    threshold_time = '2022-04-17 05:03:16'
    df_filtered = df_krill.loc[df_krill['date'] > threshold_time]
    test = 'TuC53bpfmMbRDqoEoQhvMxusCwkSGyTBpQT5WmYzK5oBBWAJBMtn3xmbQFXDyihs2TvazCaXzxiaj8TQnhXqDmz'
    # wallets which sent krill and may/may not have received embryo
    df_merge = df_krill.merge(df, left_on = 'transaction', right_on = 'transaction_id',
                              how = 'inner')
    df_filt_parents = df_merge.groupby(['nft1_name', 'nft2_name']).size().reset_index().rename({0 : 'count'}, axis = 1)
    test_mint_wallet = 'HjeQXYYd69Q8VRStiwaFnAHLcbJJRLgh9cJxpccPcLsp'
    df_test = df_merge.loc[(df_merge['wallet'] == test_mint_wallet)]
    failed_tx = df_merge['transaction'].unique().tolist()
    df_failed_tx = df.loc[~(df['transaction_id'].isin(failed_tx))]
    with open('./data/breeding/data_breeding.json', 'r') as file:
        mint_wallet_breeding = json.load(file)
    df_holders = defaultdict(list)
    holder_keys = list(holders.keys())
    for holder in holder_keys:
        value = holders[holder]
        amount = value['amount']
        df_holders['wallet_holder'] += [holder] * amount
        df_holders['mints'] += value['mints']
    df_holders = pd.DataFrame(df_holders)
    df_missing_baby = df_merge.merge(df_holders, how = 'inner',
                                     left_on = ['wallet', 'mint_id'],
                                     right_on = ['wallet_holder', 'mints'])

    # df_merge.to_csv('./data/breeding/krill_breeding_tx.csv')
    solscan_requests = 'https://public-api.solscan.io/account/splTransfers?account=9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX&offset=0&limit=2000'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    request = requests.get(url = solscan_requests, headers = headers)
    spl_transfer = request.json()
    with open('./data/breeding/baby_hashlist.json', 'r') as file:
        baby_hashlist = json.load(file)
    baby_transfers = []
    for spl_tx in spl_transfer['data']:
        token_add = spl_tx.get('tokenAddress', '')
        post_bal = int(spl_tx.get('postBalance', 0))
        pre_bal = int(spl_tx.get('preBalance',0))
        if token_add in baby_hashlist and (post_bal - pre_bal) == -1:
            baby_transfers.append((token_add, spl_tx))

    transfer_sigs = []
    errors = []
    for i, (token_add, transfer_tx) in enumerate(baby_transfers):
        if i % 20 == 0:
            print(i)
        signature = transfer_tx['signature']
        if len(signature) != 1:
            errors.append(transfer_tx)
        else:
            sig = signature[0]
            transx = http_client.get_confirmed_transaction(sig)
            transfer_sigs.append((token_add, transx))

    all_spl_post = defaultdict(list)
    errors = []
    for token_add, details in transfer_sigs:
        try:
            meta = details['result']['meta']
            post_bal = meta['postTokenBalances']
            pre_bal = meta['preTokenBalances']
            signatures = details['result']['transaction']['signatures']
            wallet = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
            pre_bal_recipient = 0
            pre_bal_sender = 0
            first_wallet_add, first_wallet_amount = pre_bal[0]['owner'], float(pre_bal[0]['uiTokenAmount']['amount'])
            # second_wallet_add, second_wallet_amount = pre_bal[1]['owner'], float(pre_bal[1]['uiTokenAmount']['amount'])

            first_wallet_add_post, first_wallet_amount_post = post_bal[0]['owner'], float(post_bal[0]['uiTokenAmount']['amount'])
            second_wallet_add_post, second_wallet_amount_post = post_bal[1]['owner'], float(post_bal[1]['uiTokenAmount']['amount'])
            signature = signatures[0]
            mint_id = post_bal[0]['mint']
            all_spl_post['transaction'].append(signature)
            all_spl_post['first_wallet'].append(first_wallet_add_post)
            all_spl_post['first_wallet_post_bal'].append(first_wallet_amount_post)
            all_spl_post['second_wallet'].append(second_wallet_add_post)
            all_spl_post['second_wallet_post_bal'].append(second_wallet_amount_post)
            all_spl_post['mint_address'].append(mint_id)
        except Exception as e:
            print(e)
            errors.append((token_add, details))

    spl_post_df = pd.DataFrame(all_spl_post)
    spl_post_df = spl_post_df.assign(diff = lambda x: x['first_wallet_post_bal'] - x['second_wallet_post_bal'])
    index = spl_post_df['diff'] > 0
    spl_post_df['recipient'] = spl_post_df['second_wallet']
    spl_post_df.loc[index, 'recipient'] = spl_post_df.loc[index, 'first_wallet']
    mint_wallet = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
    test_mint_wallet = 'ABK6KFwGksncWNii4UkqKUXXwLxHAHHxwSM4dX8LR1kv'
    spl_df = spl_post_df.loc[spl_post_df['recipient'] != test_mint_wallet]
    df_krill_final = df_krill.loc[(df_krill['sender'] != test_mint_wallet)]
    df_krill_final['amount'] = df_krill_final['first_wallet_diff'].abs()
    # df_krill_final.to_csv('./data/breeding/df_krill_transactions.csv', index = False)

    df_final = df_krill_final.merge(df, left_on='transaction', right_on='transaction_id',
                                    how='inner')
    # df_final.to_csv('./data/breeding/df_final.csv', index = False)

    current_embryo = holders[mint_wallet]['mints']
    df_combined = df_final.merge(spl_post_df, how = 'left', left_on = ['wallet', 'mint_id'],
                                 right_on = ['recipient', 'mint_address'],
                                 suffixes = ('_krill_tx', '_spl_tx'))
    df_parents_filt = df_combined.groupby(['nft1_name', 'nft2_name']).size().reset_index().rename({0 : 'count'}, axis = 1)
    duplicated_df = df_parents_filt.loc[(df_parents_filt['count'] > 1)]
    total_nan = df_combined['mint_address'].isna().sum()
    print(f'Total Number of Unsent Embryo is {total_nan}')
    mint_id_count = ((df_combined['mint_id'].isna()) & (~df_combined['mint_address'].isna())).sum()
    assigned_mints = random.sample(current_embryo, total_nan)
    df_combined['assigned_mint_id'] = df_combined['mint_address']
    nan_index = df_combined['mint_address'].isna()
    df_combined.loc[nan_index, 'assigned_mint_id'] = assigned_mints
    nunique = len(df_combined['assigned_mint_id'].unique())
    print(f'Total Number of Unique Addresses is {nunique}')

    df_refunds = df_combined.merge(duplicated_df, how = 'inner',
                                   left_on = ['nft1_name', 'nft2_name'],
                                   right_on = ['nft1_name', 'nft2_name'])

    # df_refunds.to_csv('./data/breeding/breeding_refunds.csv', index = False)
    df_combined = df_combined.sort_values(by = 'date', axis = 0, ignore_index = True)
    df_combined_unique = df_combined.drop_duplicates(subset = ['wallet', 'nft1_name', 'nft2_name'], ignore_index = True)
    df_combined_unique = df_combined_unique.sort_values(by = 'date', axis = 0, ascending = False, ignore_index = True)
    # df_combined_unique.to_csv('./data/breeding/final_data_breeding.csv', index = False)
    #
    # pvt_key = '4pBXB5AAtqXMHY27ivdMhE5xbENQ6WBB5HUsZB6XDQuANKEZQzJGAKR2uNqPXhkh7NVdB3SqDBmhvsrFMqwE8vJR'
    # pub_key = '9tG1aaVYaD76NvXBDudUeTYbHhrtyPgLX9tT42T67vwX'
    # src_wallet = SolanaWallet(
    #     secret_recovery_phrase=None,
    #     derivation_path=DerivationPath.CLI_PATH,
    #     passphrase=str(),
    #     private_key=None,  # OR You can supply this instead. e.g, [11, 234, ... 99, 24]
    #     b58_private_key=pvt_key)
    #
    # errors_transfer = []
    # owed_nft_df = df_combined_unique.loc[(df_combined_unique['mint_address'].isna())]
    # for row in owed_nft_df.iterrows():
    #     idx, row_data = row
    #     assigned_id = row_data['assigned_mint_id']
    #     recipient_address = row_data['wallet']
    #     try:
    #         transaction_signature = BLOCKCHAIN_API_RESOURCE.transfer(
    #             wallet=src_wallet,
    #             recipient_address=recipient_address,
    #             token_address=assigned_id,
    #             network=SolanaNetwork.MAINNET_BETA
    #         )
    #     except Exception as e:
    #         print(e)
    #         errors_transfer.append(row)


    # errors_transfer_two = []
    # owed_nft_df = df_combined_unique.loc[(df_combined_unique['mint_address'].isna())]
    # for i, row in enumerate(errors_transfer):
    #     if i % 10 == 0:
    #         print(i)
    #     idx, row_data = row
    #     assigned_id = row_data['assigned_mint_id']
    #     recipient_address = row_data['wallet']
    #     try:
    #         transaction_signature = BLOCKCHAIN_API_RESOURCE.transfer(
    #             wallet=src_wallet,
    #             recipient_address=recipient_address,
    #             token_address=assigned_id,
    #             network=SolanaNetwork.MAINNET_BETA
    #         )
    #         print(f'completed {i} : {transaction_signature}')
    #     except Exception as e:
    #         print(e)
    #         errors_transfer_two.append(row)
    #
    # cant_breed_whales = df_combined_unique[['nft1_name', 'nft2_name']]
    # cant_breed_whales.to_csv('./data/breeding/whales_that_cant_breed.csv', index = False)