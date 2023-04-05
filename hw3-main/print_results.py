import transformers
import pickle 
import os 

PATH =  "/gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main/outputs/"
for filename in [
	'train_results.bitfit.04-02-2023.15-35-57.pickle', 
	'train_results.no-bitfit.04-02-2023.15-33-21.pickle',
	'train_results.bitfit.04-03-2023.03-38-15.pickle', 
	'train_results.no-bitfit.04-03-2023.04-57-17.pickle',
	'checkpoint.no-bitfit.04-02-2023.15-33-21.test_results.p', 
	'checkpoint.bitfit.04-02-2023.15-35-57.test_results.p',
]:
	with open(os.path.join(PATH, filename) , 'rb') as f:
		results = pickle.load(f)
		print(filename, results)
