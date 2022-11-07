import pandas as pd
import numpy as np
import time
import os
import httpx

def auth():
    return os.getenv('TOKEN')

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(end_point):
    return "https://api.twitter.com/2/tweets"+end_point #Change to the endpoint you want to collect data from

def aync_connect_to_endpoint(url, headers, params, next_token = None):
    if next_token is not None:
        params['next_token'] = next_token   #params object received from create_url function
    response = httpx.get(url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

if __name__ == '__main__':
	os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAP4KfQEAAAAA%2BdXC4M0Cdomai4mZbMOk%2BgfdWyI%3D2V3mXGjkESDNGYB9ARMZMu8zfELNfbmtYWv6l0BDx1e8g8pQXx'
	bearer_token = auth()
	headers = create_headers(bearer_token)

	# key:   magiceden collection
	# value: Twitter query for collection
	keywords = {"aurory": "@AuroryProject",
				"degods": "@DeGodsNFT",
				"smb": "@SolanaMBS",
				"cetsoncreck": "@CetsOnCreck",
				"degenapes": "@DegenApeAcademy",
				"blocksmithlabs": "@blocksmithlabs",
				"stonedapecrew": "@StonedApeCrew",
				"taiyorobotics": "@TaiyoRobotics",
				"solgods": "@TheFracture_",
				"boryokudragonz": "@BoryokuDragonz",
				"famousfoxfederation": "@FamousFoxFed",
				"catalinawhalemixer": "@catalinawhales",
				"galacticgeckos": "@GalacticGeckoSG",
				"thugbirdz": "@thugbirdz",
				"mindfolk": "@mindfolkART",
				"nyanheroes": "@nyanheroes",
				"tombstonedhighsociety": "@TombStonedHS",
				"female_hodl_whales": "@SolanaWhalesNFT"}

	date_range = pd.date_range(start="2021-01-31",end="2022-08-01", freq="M")
	print(date_range)

	df_result = pd.DataFrame({"end_date": pd.date_range(start="2021-01-31",end="2022-07-31", freq="H")})
	print(df_result)
	print(df_result.shape)

	for key, value in keywords.items():
		count_data = []

		counts_all_params = {'query': value}
		counts_all_params["granularity"] = "hour"
		counts_all = "/counts/all"

		for i in range(1, len(date_range)):
			counts_all_url = create_url(counts_all)
			print(counts_all_url)
			counts_all_params["start_time"] = date_range[i-1].strftime('%Y-%m-%dT%H:%M:%SZ')
			counts_all_params["end_time"] = date_range[i].strftime('%Y-%m-%dT%H:%M:%SZ')

			print(counts_all_params)

			json_response = aync_connect_to_endpoint(counts_all_url, headers, counts_all_params)
			count_data.extend(json_response["data"])
			print(json_response["meta"])
			while "meta" in json_response and "next_token" in json_response["meta"] and json_response["meta"]["next_token"] is not None:
				json_response = aync_connect_to_endpoint(counts_all_url, headers, counts_all_params, json_response["meta"]["next_token"])
				print(json_response["meta"])
				
			time.sleep(5)

		df_count = pd.DataFrame(count_data)
		df_result[key] = df_count["tweet_count"]
		df_result = df_result.dropna()
		df_result[key] = df_result[key].astype(dtype='int')
		#df_count.to_feather(str('data/'+key+'_tweet_count.feather'))
	df_result.to_feather(str('data/collection_tweet_count_hourly.feather'))
