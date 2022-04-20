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
const map = new Map();

async function getAddressNFTs(address) {
  try {
    // const address = "3EqUrFrjgABCWAnqMYjZ36GcktiwDtFdkNYwY6C6cDzy";
    // or use Solana Domain
    // const address = "NftEyez.sol";
    // const address = jsonWallet[index];

    const getPublicAddressConst = async function getPublicAddress() {
      const publicAddress = await resolveToWalletAddress({
        address
      });

      // console.log("publicAddress", publicAddress);

      return publicAddress;
    };

    
    const publicAddress = await getPublicAddressConst();
    const rpcUrl =
      "https://polished-cool-surf.solana-mainnet.quiknode.pro/66bc31e049b46dd293092114e2d8717b3378033f/";
    const connection = new Connection(rpcUrl, "confirmed");
    const nArray = await getParsedNftAccountsByOwner({
      publicAddress,
      connection,
    })
    console.log(nArray)
      // console.log("abc", publicAddress);
      // console.log("setting", publicAddress);
      // map.set(publicAddress, nArray);
      // console.log([...map.keys()]);
      // console.log("map", map);

    // return [publicAddress, nArray];
    

    // return getNftArrayConst();
  } catch (e) {
    console.log("Error thrown, while fetching NFTs", e.message);
  }
}
const test = async () => {
  for (const key of Object.values(jsonWallet)) {
    console.log(`${key} yo`);
    const test = key;
    let response = await getAddressNFTs(key);
    console.log(response)
  }
}
test()

      // const getPublicAddressConst = async () => {
      // let response = await getAddressNFTs(key);
      // // map.set(publicAddress, nArray);
      // // console.log(response);
      // map.set(response?.[0], response?.[1]);
  // getPublicAddressConst();
  // const [publicAddress, nArray] = await getPublicAddressConst();
  // console.log(getPublicAddressConst());
  // console.log([...map.entries()]);


// setTimeout(() => {
//   console.log([...map.entries()]);
// }, 20000);
// var obj = Objecst.fromEntries(map);
// var jsonobj = JSON.stringify(obj);
// console.log(obj);
// map.set(1, "hello");