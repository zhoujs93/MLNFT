import {
  resolveToWalletAddress,
  getParsedNftAccountsByOwner,
} from "@nfteyez/sol-rayz";
import { Wallet } from "@project-serum/anchor";
import { clusterApiUrl, Connection, PublicKey, Keypair } from "@solana/web3.js";
import jsonWallet from "./degods_wallet.json";
var fs = require('fs');

// console.log(jsonWallet);
var n = jsonWallet.length;
var index = 0;
const map = new Map();

async function getAddressNFTs(address: string) {
  try {

    const getPublicAddressConst = async function getPublicAddress() {
      const publicAddress = await resolveToWalletAddress({
        text: address,
      });

      // console.log("publicAddress", publicAddress);

      return publicAddress;
    };
    const getNftArrayConst = async function getNftArray() {
      const publicAddress = await getPublicAddressConst();
      const rpcUrl =
        "https://polished-cool-surf.solana-mainnet.quiknode.pro/66bc31e049b46dd293092114e2d8717b3378033f/";
      const connection = new Connection(rpcUrl, "confirmed");
      const nArray = await getParsedNftAccountsByOwner({
        publicAddress,
        connection,
      });

      return [publicAddress, nArray];
    };

    return getNftArrayConst();
  } catch (e) {
    console.log("Error thrown, while fetching NFTs", (e as Error).message);
  }
}

for (const key of Object.values(jsonWallet)) {
  const getPublicAddressConst = async () => {
    let response = await getAddressNFTs(key);
    // map.set(publicAddress, nArray);
    // console.log(response);
    map.set(response?.[0], response?.[1]);
  };
  getPublicAddressConst();
}

setTimeout(() => {
  console.log([...map.entries()]);
  var obj = Object.fromEntries(map);
  var jsonobj = JSON.stringify(obj);
  var myEscapedJSONString = jsonobj.replace(/\\n/g, "\\n")
                                   .replace(/\\'/g, "\\'")
                                   .replace(/\\"/g, '\\"')
                                   .replace(/\\&/g, "\\&")
                                   .replace(/\\r/g, "\\r")
                                   .replace(/\\t/g, "\\t")
                                   .replace(/\\b/g, "\\b")
                                   .replace(/\\f/g, "\\f");
  fs.writeFile ("./degods_wallet_holdings.json", myEscapedJSONString, function(err: any) {
    if (err) throw err;
    console.log('complete');
    }
);
}, 20000);

