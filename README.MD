# Introduction

This project focus on the scenario that a startup company with available thirty days data on their database, and the company want to improve their customer satisfaction by deploying a new recommendation system in their website. This main part of this project is the implementation of the algorithms on recommender system, including item-based collaborative filtering, user-based collaborative filtering, demographic filtering, content-based recommendation. The goal of this project is to make a comparison between several algorithms and make further exploration on the algorithm fusion.

# System architecture

There are three main parts that consisted by this recommender system: online platform, offline platform and algorithms tools. Algorithms tools is the existed python packages for recommendations. The output of offline platform is stored into mongo database. The online part will take advantage of the pre-calculated offline data and make recommendations. The effectiveness of this system is only evaluated offline. The data source will be the user data, item data and the interaction data on the left hand-side of the following graph.

<div align=center><img alt="System architecture" src="https://github.com/great-expectations/great_expectations/raw/develop/generic_dickens_protagonist.png"/></div>

For the online platform, it mainly uses java scripts, which load the stored output from offline part, and make further computation based on these offline pre-calculated mapping. The mapping can also be considered as a model or recommendation list for each user or item.

For the offline platform, there are two main categories: user to items mapping table and item to items mapping table. For the user to items mapping table, the system will need to give a user ID, and the system will return the top 10 items as its recommendation, which is the items that user probably tend to click from the method aspect. It is similar for the item to items case, the system only need be provided an item ID, and it will return the top 10 items as its recommendation.

# Technical solutions

The following graph is the data follows of this system, which also contains technical details.

<div align=center><img alt="Architecture detail" src="https://github.com/great-expectations/great_expectations/raw/develop/generic_dickens_protagonist.png"/></div>

This system uses click interaction as primary target to extract information. Joined the tables to get the desired table shown as follows, which contains SKU ID, user ID. Each row is a click by the user. (Preprocess) It removed click records with the same user ID and request time since users may click several times on their screen, but the effective records in there are only once. For privacy protection, the SKU ID and user ID are encoded, which make the result of the system not that intuitive.

<div align=center><img alt="joined tab" src="https://github.com/great-expectations/great_expectations/raw/develop/generic_dickens_protagonist.png"/></div>


For collaborative filtering, the offline platform of this system retrieved accumulated click counts for each user by SKU ID to get a user item matrix first, which is kind of panel data for user ID and SKU ID. For simulation the true environment, consider that it only has 15 days data, and the system will update day by day with the time moving forward until the end of this dataset. The user item matrix is accumulated from beginning since users probably have more inclination to buy the item if they click more times of that item. With this matrix, this system implemented the user based collaborative filtering and item based collaborative filtering with the Pearson and cosine similarity measurements.

For content-based recommendation, the algorithm get the grouping table for item attributes first, then it finds the 10 highest click frequency items in each group, and finally join the table back with item attributes. In other words, the algorithm group items based on their attributes first, and find what is the most popular ten items in each group. If a user clicks an item within one group, the system will just recommend the most popular items in that group, on the recent 7 days. Obviously, the more attributes the items had, the more accuracy this method will be.

For demographic based filtering, it is almost the same as previous one. The only difference is that we need to grouping table by user attributes first.

For item embedding, the algorithm first generate user click sequence, and treat these SKU ID sequence as vocabulary in NLP concept, then get the embedding for each SKU ID by training a skip-gram model with genism package in python.

The ranking part of this system is computed based on the scores on the scoring part, rather than ranking neural networks. The summary is shown as graph as follows.


<div align=center><img alt="Offline platform" src="https://github.com/great-expectations/great_expectations/raw/develop/generic_dickens_protagonist.png"/></div>


# Experiments

The [dataset](https://www.dropbox.com/sh/kdgx4wconscfllj/AADNR94dAP8PlmGCYhHLT5Eqa?dl=0) only contains 30 successive days records, which contains 31868 unique SKU ID, 457298 unique user ID and 20214515 click records. The interaction data is about 1 GB, but when loading into memory and computing the correlations, the RAM will be consumed really fast. To speed up experiment, this project only randomly choosing about 10% users and items as target first. As show on the technical solutions part, the SKU table contains two attributes, which are encoded. The user table contains 9 attributes, which are listed as follows.


 <div align=center><img alt="User table attributes" src="https://github.com/great-expectations/great_expectations/raw/develop/generic_dickens_protagonist.png"/></div>



For environment, this experiment implemented on windows system, which has 32 GB RAM and AMD 2700X as CPU.

For evaluation, A/B test is obviously not available in this dataset on this period since there is no influenced user-item interaction generated by this system. Therefore, this project only evaluated on daily scale based on accuracy. For item to items method, given previous click SKU ID, the system will predict next click of SKU ID. For user to items method, given user ID, the system will predict next click of SKU ID. And the following [table](https://www.dropbox.com/sh/kdgx4wconscfllj/AADNR94dAP8PlmGCYhHLT5Eqa?dl=0) make the summary and comparison between different algorithms.

Rec_num is the parameter that decides how many SKU that will recommend to the user. Basically, the more recommended number of items, the higher accuracy the algorithm will have. Because it will cover more possible item that user probably click in next time.

# Conclusion

From comparison on several days, the item-based recommendation (which directly calculate the cosine similarity between different items based on user item matrix) and the demographic based recommendation are more effective than other methods in there.

The online model fusion part should be only written in java scripts form, which is not implemented in this period. Also, the evaluation method such as A/B test for this dynamic online fusion method is needed.

In the future, this project will be tested on different dataset that without encoding the SKU and user ID, which may have more intuitive recommendation result. Furthermore, with more computational resource, this system should be run on the large scale with the prepared parameters in functions. On ranking model aspect, to further improve the accuracy, this project probably extends more neural network models. Of course, the speed of the response will need to be considered.
