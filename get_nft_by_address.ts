import {
    resolveToWalletAddress,
    getParsedNftAccountsByOwner,
  } from "@nfteyez/sol-rayz";
  
  try {
    const address = "Gnv65FfQHthe7PLj3TSfgbmKjB5siCiq4Kk5m3SKXyBw;
    // or use Solana Domain
    // const address = "NftEyez.sol";
  
    const publicAddress = await resolveToWalletAddress({
      text: address
    });
  
    const nftArray = await getParsedNftAccountsByOwner({
      publicAddress,
    });
    console.log(nftArray)
  } catch (error) {
    console.log("Error thrown, while fetching NFTs", error.message);
  }