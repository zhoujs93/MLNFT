// import { programs } from '@metaplex/js';
var metaplex = require('@metaplex/js')
var _ = require('lodash');
const web3 = require("@solana/web3.js");
const bs58 = require('bs58')
var fs = require('fs');
const axios = require('axios');
var PublicKey = web3.PublicKey
const sendAndConfirmTransaction = web3.sendAndConfirmTransaction
const Keypair = web3.Keypair
var programs = metaplex.programs
var uploader = require("../uploader/uploader.js")
// console.log(process.env)
var keyfilename = "../keypairs/"+ process.env.KP_NAME
var keyfile = require(keyfilename)
exports.updateMetadataV1 = async (mintkey, solConnection) => {
  let { metadata : {Metadata, UpdateMetadata, MetadataDataData, Creator} } = programs;
  try {
    console.log("UPDATING METADATA")
    let nftMintAccount = new PublicKey(mintkey);
    let signer = Keypair.fromSecretKey(new Uint8Array(keyfile));
    let metadataAccount = await Metadata.getPDA(nftMintAccount);
    const metadat = await Metadata.load(solConnection, metadataAccount);
    var metadata_uri = metadat.data.data.uri
    var metadata = null
    var resp = await axios.get(metadata_uri)
    var old_metadata = resp.data
    var new_metadata = Object.assign({}, old_metadata)
    var attrs = [{ trait_type: 'Type', value: 'Embryo' }]
    new_metadata["attributes"] = attrs
    // console.log("UPDATING METADATA")
    
    new_metadata["external_url"] = "https://solanahodlwhales.io"
    // console.log(new_metadata)
    var files = new_metadata["properties"]["files"] 
    var new_uploaded_metadata_url = await uploader.uploadJsonToArweave(new_metadata)
    if (metadat.data.data.creators != null) {
      const creators = metadat.data.data.creators.map(
        (el) =>
            new Creator({
                ...el,
            }),
      );
      let newMetadataData = new MetadataDataData({
        name: metadat.data.data.name,
        symbol: metadat.data.data.symbol,
        uri: new_uploaded_metadata_url,
        creators: [...creators],
        sellerFeeBasisPoints: metadat.data.data.sellerFeeBasisPoints,
      })
      const updateTx = new UpdateMetadata(
        { feePayer: signer.publicKey },
        {
          metadata: metadataAccount,
          updateAuthority: signer.publicKey,
          metadataData: newMetadataData,
          newUpdateAuthority: signer.publicKey,
          primarySaleHappened: metadat.data.primarySaleHappened,
        },
      );
      let result = await sendAndConfirmTransaction(solConnection, updateTx, [signer]);
      // console.log("result =", result);
      console.log("Metadata Updated for NFT", mintkey);
    }
  } catch (error) {
    console.log(error)
    throw new Error("error")
    // console.log("====================== ERROR ======================")
  }
}