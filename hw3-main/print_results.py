import transformers
import pickle 
import os 

PATH =  "/gpfs/home/kgz2437/pathology_parsing/testing/NLU/Natural-Language-Understanding/hw3-main/outputs/"
for filename in [
"train_results.bitfit.04-06-2023.01-26-58.pickle", 
"train_results.no-bitfit.04-06-2023.18-46-38.pickle",
"train_results.no-bitfit.04-06-2023.18-47-32.pickle",
"train_results.no-bitfit.04-06-2023.21-23-34.pickle",
"train_results.bitfit.04-06-2023.21-19-03.pickle",
"checkpoint.bitfit.04-06-2023.21-19-03.test_results.p", 
"checkpoint.no-bitfit.04-06-2023.21-23-34.test_results.p",
]:
	with open(os.path.join(PATH, filename) , 'rb') as f:
		results = pickle.load(f)
		print(filename, results)
