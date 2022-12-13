import pandas as pd
import numpy as np

#构造数据
df=pd.DataFrame({'a':['?',7499,'?',7566,7654,'?',7782],'b':['SMITH', '.','$','.' ,'MARTIM','BLAKE','CLARK'],
'c':['CLERK','SALESMAN','$','MANAGER','$','MANAGER','$'],
'd':[7902,7698,7698,7839,7698,7839,7839],
'e':['1980/12/17','1981/2/20','1981/2/22','1981/4/2','1981/9/28','1981/5/1','1981/6/9'],
'f':[800,1600,1250,2975,1230,2859,2450],
'g':[np.nan,300.0,500.0,np.nan,1400.0,np.nan,np.nan],
'h':[20,30,30,20,30,30,10]})
print(df)

#替换全部或者某行某列
#全部替换，这二者效果一样
df.replace(20,30, inplace=True)
# df.replace(to_replace=20,value=30)
print(df)

# #某一列或者某几列
# df['h'].replace(20,30)
# df[['b','c']].replace('$','rmb')
#
# #某一行或者几行
# df.iloc[1].replace(1600,1700)
# df.iloc[1:3].replace(30,40)
#
# #inplace=True
# df.replace(20,30,inplace=True)
# df.iloc[1:3].replace(30,40,inplace=True)
#
#
# #用list或者dict进行单值或者多值填充,
# #单值
# #注意，list是前者替换后者，dict字典里的建作为原值，字典里的值作为替换的新值
# df.replace([20,30])
# df.replace({20:30})
# #多值,list是list逗号后的值替换list的值，dict字典里的建作为原值，字典里的值作为替换的新值
# df.replace([20,1600],[40,1700])  #20被40替换，1600被1700替换
# df.replace([20,30],'b')  #20,30都被b替换
# df.replace({20:30,1600:1700})
# df.replace({20,30},{'a','b'})  #这个和list多值用法一样
#
# #,method
# #其实只需要传入被替换的值，
# df.replace(['a',30],method='pad')
# df.replace(['a',30],method='ffill')
# df.replace(['a',30],method='bfill')
#
# #可以直接这样表达
# df.replace(30,method='bfill')  #用30下面的最靠近非30的值填充
# df.replace(30,method='ffill')  #用30上面最靠近非30的值填充
# df.replace(30,method='pad')   #用30上面最靠近非30的值填充
#
# #一般用于空值填充
# df.replace(np.nan,method='bfill')
#
# #limit
# df.replace(30,method='bfill',limit=1)  #现在填充的间隔数