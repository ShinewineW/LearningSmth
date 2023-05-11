"""
Wind Statistics
----------------

Topics: Using array methods over different axes, fancy indexing.

1. The data in 'wind.data' has the following format::

        61  1  1 15.04 14.96 13.17  9.29 13.96  9.87 13.67 10.25 10.83 12.58 18.50 15.04
        61  1  2 14.71 16.88 10.83  6.50 12.62  7.67 11.50 10.04  9.79  9.67 17.54 13.83
        61  1  3 18.50 16.88 12.33 10.13 11.17  6.17 11.25  8.04  8.50  7.67 12.75 12.71

   The first three columns are year, month and day.  The
   remaining 12 columns are average windspeeds in knots at 12
   locations in Ireland on that day.

   Use the 'loadtxt' function from numpy to read the data into
   an array.

2. Calculate the min, max and mean windspeeds and standard deviation of the
   windspeeds over all the locations and all the times (a single set of numbers
   for the entire dataset).

3. Calculate the min, max and mean windspeeds and standard deviations of the
   windspeeds at each location over all the days (a different set of numbers
   for each location)

4. Calculate the min, max and mean windspeed and standard deviations of the
   windspeeds across all the locations at each day (a different set of numbers
   for each day)

5. Find the location which has the greatest windspeed on each day (an integer
   column number for each day).

6. Find the year, month and day on which the greatest windspeed was recorded.

7. Find the average windspeed in January for each location.

You should be able to perform all of these operations without using a for
loop or other looping construct.

Bonus
~~~~~

1. Calculate the mean windspeed for each month in the dataset.  Treat
   January 1961 and January 1962 as *different* months. (hint: first find a
   way to create an identifier unique for each month. The second step might
   require a for loop.)

2. Calculate the min, max and mean windspeeds and standard deviations of the
   windspeeds across all locations for each week (assume that the first week
   starts on January 1 1961) for the first 52 weeks. This can be done without
   any for loop.

Bonus Bonus
~~~~~~~~~~~

Calculate the mean windspeed for each month without using a for loop.
(Hint: look at `searchsorted` and `add.reduceat`.)

Notes
~~~~~

These data were analyzed in detail in the following article:

   Haslett, J. and Raftery, A. E. (1989). Space-time Modelling with
   Long-memory Dependence: Assessing Ireland's Wind Power Resource
   (with Discussion). Applied Statistics 38, 1-50.


See :ref:`wind-statistics-solution`.
"""



from numpy import loadtxt
import numpy as np


#1. Solution 1. load data
print("-----------------------Below is Solution1-----------------------")
data = loadtxt('C:/Users/Administrator/Desktop/MyGit/LearningSmth/PythonLearning/exercises/wind_statistics/wind.data')
print(data.shape)


#2.Soluton2. 计算整个数据集中风速的最大，最小，平均值和标准差
print("-----------------------Below is Solution2-----------------------")
data_2 = data[:,3::]
print(data_2.shape)

#计算最大值
print("The WindSpeed Max Value:" + str(np.max(data_2)))
print("The WindSpeed Min Value:" + str(np.min(data_2)))
print("The WindSpeed Mean Value:" + str(np.mean(data_2)))
print("The WindSpeed standard deviation Value:" + str(np.std(data_2)))

#3.Solution3. 计算每个地方全时间范围内的最大最小平均标准差
#但是直接用指定轴就不用使用for循环了，如Solution4的方法
print("-----------------------Below is Solution3-----------------------")
for i in range(3,15,1):
    data_3 = data[:,i]
    print("The  " + str(i) +" th location Windspeed situation:")
    print("The WindSpeed Max Value:" + str(np.max(data_3)))
    print("The WindSpeed Min Value:" + str(np.min(data_3)))
    print("The WindSpeed Mean Value:" + str(np.mean(data_3)))
    print("The WindSpeed standard deviation Value:" + str(np.std(data_3)))
#如下是非loop循环版本
print(data_2.max(axis=0))
print(data_2.min(axis=0))
print(data_2.mean(axis=0))
print(data_2.std(axis=0))   
    
    
#4.Solution4.计算所有地方每一天的风速情况
print("-----------------------Below is Solution4-----------------------")
print("不要尝试直接输出")
print(data_2.max(axis=-1).shape)
print(data_2.min(axis=-1).shape)
print(data_2.mean(axis=-1).shape)
print(data_2.std(axis=-1).shape)

#5.Solution5.计算所有地方每一天的风速最大存在地点
print("-----------------------Below is Solution5-----------------------")
data_5 = data_2.argmax(axis=-1) + 3
print(data_5.shape)

#6.Solution6. 找到最大风速存在地点的年月日
print("-----------------------Below is Solution6-----------------------")
#Way1 直接根据库函数根据argmax返回的索引直接得到具体的坐标值
SpeedMax_Index = np.unravel_index(np.argmax(data_2),data_2.shape)
print(SpeedMax_Index)
print(data[SpeedMax_Index[0],0:3:1])
#Way2 首先对所有的横行求max，然后在对整列求argmax这样就会返回你需要的某行数据
row_index  = np.argmax(data_2.max(axis = -1))
print(row_index)
print(data[row_index,0:3:1])
#Way3 mask方法
row_index = 0
row_index , col_index = np.where(data_2 == data_2.max())
print(data[row_index,0:3:1])

#7.Solution7. 找到每个地方1月的风速平均值
print("-----------------------Below is Solution7-----------------------")
data_time = data[:,1]
mask_Jan = data_time == 1
data_Jan = data_2[mask_Jan] #此处使用了numpy中的广播规则，也就是只要是行列中某一项对齐，numpy会广播其他维度进行计算
print(data_Jan.shape)
print(data_Jan.mean(axis = 0))

#Bonus. Solution1 找到数据集中每个月的风速平均值
print("-----------------------Below is Bonus Solution1-----------------------")
data_Year = data[:,0]
data_Month = data[:,1]
#根据年月生成每个月份独一无二的数字 从61年1月开始 计算为0
Months = (data_Year - 61) * 12 + data_Month - 1
print(Months)
#Months = Months.
Months = Months.astype(int)
Months_unique_Value = set(Months) #转成不重复的列表 哈希表这种结构
print(len(Months_unique_Value))
Monthly_Means = np.zeros(len(Months_unique_Value)) #构建存储结果的矩阵
for i in Months_unique_Value:
    Monthly_Means[i] = data_2[Months == i].mean()
print(Monthly_Means)

#Bonus. Solution2 计算每个礼拜所有地方的风速情况
print("-----------------------Below is Bonus Solution2-----------------------")
data_Fisrt_Year_All_Week = data_2[:52*7,:].reshape(-1,7*12)
#此处reshape的用法，-1表示我不确定第一个的维度，根据后一个参数自动计算前一个的维度
print(data_Fisrt_Year_All_Week.shape)

print('  min:', data_Fisrt_Year_All_Week.min(axis=1))
print('  max:', data_Fisrt_Year_All_Week.max(axis=1))
print('  mean:', data_Fisrt_Year_All_Week.mean(axis=1))
print('  standard deviation:', data_Fisrt_Year_All_Week.std(axis=1))

#BonuesBonuse. Solution1. BonusSolution1中的进阶版，如何可以得到不用for循环的每个月平均值
#过于tricky解答过程可以参考soltuion.py文件中的办法。







