import os
from tqdm import tqdm



if __name__ == '__main__':
    for day in tqdm(range(15, 30)):
        os.system("python .\pydata\submain_daily_update.py -t 2018-03-"+str(day) + " --no-ubcf")

        
