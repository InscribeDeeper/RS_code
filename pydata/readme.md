# preprocessing module
- Since we only use once for the preprocessing module, we dont have to write the code into class format.
- This part of code should be tested on jupyter notebook for convenience.

CLI
- python .\pydata\system_init.py .\pydata\main_daily_update.py
- python .\pydata\main_daily_update.py -t 2018-03-14
- python .\pydata\main_daily_update.py -t 2018-03-15
- python .\pydata\main_daily_update.py -t 2018-03-16


1. main_daily_update will generate the cum_ui_mtx for next_date 
2. On next date, user2item and item2item method will load the next_date cum_ui_mtx to generate the middle_data and saved into DB