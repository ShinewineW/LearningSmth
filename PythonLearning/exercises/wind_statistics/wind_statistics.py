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
data = loadtxt('C:/Users/Administrator/Desktop/MyGit/LearningSmth/PythonLearning/exercises/wind_statistics/wind.data')
print(data.shape)


#2.Soluton2. 计算整个数据集中风速的最大，最小，平均值和标准差
data_2 = data[:,3::]
print(data_2.shape)

#计算最大值
print("The WindSpeed Max Value:" + str(np.max(data_2)))
print("The WindSpeed Min Value:" + str(np.min(data_2)))
print("The WindSpeed Mean Value:" + str(np.mean(data_2)))
print("The WindSpeed standard deviation Value:" + str(np.std(data_2)))

#3.Solution3. 计算每个地方全时间范围内的最大最小平均标准差
#但是直接用指定轴就不用使用for循环了，如Solution4的方法
for i in range(3,13,1):
    data_3 = data[:,i]
    print("The  " + str(i) +" th location Windspeed situation:")
    print("The WindSpeed Max Value:" + str(np.max(data_3)))
    print("The WindSpeed Min Value:" + str(np.min(data_3)))
    print("The WindSpeed Mean Value:" + str(np.mean(data_3)))
    print("The WindSpeed standard deviation Value:" + str(np.std(data_3)))
    
    
    
#4.Solution4.计算所有地方每一天的风速情况
print(data_2.max(axis=-1).shape)
print(data_2.min(axis=-1).shape)
print(data_2.mean(axis=-1).shape)
print(data_2.std(axis=-1).shape)

#5.Solution5.计算所有地方每一天的风速最大存在地点
data_5 = data_2.argmax(axis=-1) + 3
print(data_5.shape)

#6.Solution6. 找到最大风速存在地点的年月日
#print(data_2.argmax()//12)
#print(np[(data_2.argmax()//12)::,0:3:1])






