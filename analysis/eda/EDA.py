
# # EDA
# In[ ]:
import seaborn as sns
sns.set_style("whitegrid")
all_rec = load_daily_data(now="2018-04-1", date_field='request_time', file_path=PATH_CLICK, Filter=False, from_start=True)
group_rec = all_rec.groupby("sku_ID").count()['user_ID']
group_rec.sort_values().plot()
