// import {
//   resolveToWalletAddress,
//   getParsedNftAccountsByOwner,
// } from "@nfteyez/sol-rayz";
// import { Wallet } from "@project-serum/anchor";
// import {clusterApiUrl, Connection, PublicKey, Keypair} from "@solana/web3.js";
// import jsonWallet from "./degods_wallet.json";
var {
  resolveToWalletAddress,
  getParsedNftAccountsByOwner,
} = require("@nfteyez/sol-rayz");
var { Wallet } = require("@project-serum/anchor");
var {clusterApiUrl, Connection, PublicKey, Keypair} = require("@solana/web3.js");
var jsonWallet = require("./degods_wallet.json");

// console.log(jsonWallet);
var n = jsonWallet.length;
var index = 0;
var test = [];

while (index < n) {
  try {
    const address = "3EqUrFrjgABCWAnqMYjZ36GcktiwDtFdkNYwY6C6cDzy";
    // or use Solana Domain
    // const address = "NftEyez.sol";
    // const address = jsonWallet[index];
    const getPublicAddressConst = async function getPublicAddress() {
      const publicAddress = await resolveToWalletAddress({
        text: address,
      });
  
      // console.log("publicAddress", publicAddress);
  
      return publicAddress;
    };
    // getPublicAddress();
    const getNftArrayConst = async () => {
      const publicAddress = await getPublicAddressConst();
      const rpcUrl = 'https://polished-cool-surf.solana-mainnet.quiknode.pro/66bc31e049b46dd293092114e2d8717b3378033f/';
      const connection = new Connection(rpcUrl, 'confirmed');
      const nArray = await getParsedNftAccountsByOwner({
        publicAddress,
        connection
      });
      // console.log(nArray);
      // console.log(address)
      // map.set(address, nArray);
      // console.log([...map.keys()])
      // map[address] = nArray
      return nArray
    };
    const value = getNftArrayConst();
    console.log(value);
    index = index + 1;
  
  } catch (e) {
    // console.log("Error thrown, while fetching NFTs", e.message);
    index = index - 1;
  }
}
// var obj = Object.fromEntries(map)
// var jsonobj = JSON.stringify(obj)
// console.log('hi')
// // console.log(map);
// map.set('1', '1');
console.log(test);

// console.log([...map.values()])
