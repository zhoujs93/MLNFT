from requestData import *
from theblockchainapi import TheBlockchainAPIResource, SolanaCurrencyUnit, SolanaNetwork, SolanaWallet, DerivationPath





if __name__ == '__main__':
    pvt_key = '3kdwTTJr742wAnK5NLbwZktyzmhthDJQmLuBGYXi7zsXSgAZPTigEZfRdStYBWSnatiSpwfkx3FrqLwr3JACjwxV'
    wallet_new = SolanaWallet(
        secret_recovery_phrase=None,
        derivation_path=DerivationPath.CLI_PATH,
        passphrase=str(),
        private_key=None,  # OR You can supply this instead. e.g, [11, 234, ... 99, 24]
        b58_private_key=pvt_key)

    mint_id = '8qvtgWuJaSpWAQ4GsBFeGEe13wwgp43NkP9ATJpcDYWN'
    candy_machine_id =BLOCKCHAIN_API_RESOURCE.get_candy_machine_id_from_nft(
            mint_address=mint_id,
            network=SolanaNetwork.MAINNET_BETA
        )
