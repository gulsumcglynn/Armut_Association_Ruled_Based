# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import datetime as dt

#GÖREV1: Veriyi Hazırlama

# Adım 1: armut_data.csv dosyasınız okutunuz.

df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()
df.head()
df.info()
df.describe()

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["Hizmet"] = df["ServiceId"].astype(str)+"_"+df["CategoryId"].astype(str)

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini kullanıcı bazında "_" ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["CreateDate"].dt.year.astype(str)+"_"+df["CreateDate"].dt.month.astype(str)
df["Sepet_ID"] = df["UserId"].astype(str)+"_"+df["New_Date"].astype(str)

# GÖREV 2: Birliktelik Kuralları Üretiniz

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

invoice_product_df = df.groupby(['Sepet_ID', 'Hizmet'])["Hizmet"].count().unstack().fillna(0).\
           applymap(lambda x: 1 if x>0 else 0)

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)
frequent_itemsets.head()

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)
rules.head()

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz..
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 3)
