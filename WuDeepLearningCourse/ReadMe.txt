@Title: 吴恩达深度学习课程作业使用指南
@Author: Netfather
@Time: 2021年1月18日

1. 使用git将整个LearningSmth代码下载下来
2. 在下载完成后，其中存在一个名为 WuDeepLearningCourse的文件夹
3. 将工作目录设定为这个文件夹即可使用所有代码(相关课程中的Dataset路径已经完成修改
4. 文件命名格式为 
	Cm_Wn_Homework_Partx:  m表示是第一次课程，一共有5次课程，其中第三次课程是优化机器学习参数，没有相关的作业
				                 n表示是这一次课程中第几周的作业
							           x表示这一周的作业分为了几个部分，例如有的周有3部分作业，因此有Part1，2，3
	Cm_Wn_HomeWork_Partx_DataSet: 和上面的作业是一一对应关系，但是课程前期，一周作业尽管分为了几部分但是，会对应一个数据集，而没有Partx



最后，希望大家都能从吴恩达的深度学习课程中学习到自己想知道的知识

工程所使用的包名单，放置在本目录的 environment.yml 文件中，使用时在确保自己正确安装cuda之后，使用 conda env create -f environment.yml 命令就可以安装所有的包。
