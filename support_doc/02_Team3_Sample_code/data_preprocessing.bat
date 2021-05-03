python join_table.py
cd ./data
data_preprocessing.exe 2 sku_ID click_order_user.csv user_n_gram.csv user_level gender education city_level purchase_power marital_status age
data_preprocessing.exe 2 user_ID click_order_sku.csv sku_n_gram.csv sku_ID if_order
cd ..