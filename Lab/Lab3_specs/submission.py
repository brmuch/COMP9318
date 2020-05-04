## import modules here
import numpy as np
################# Question 1 #################
def dot_product(a, b):
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res
    
def hc(data, k):# do not change the heading of the function
    categories = [i for i in range(0, len(data))]
    
    while len(set(categories)) > k:                              # when # of cluster larger than k
        cur_categories = list(set(categories))
        proximity = np.zeros((len(cur_categories), len(cur_categories)), dtype=np.float)
        
        for i in range(0, proximity.shape[0] - 1):               # calculate the distance between categroy i and j
            for j in range(i + 1, proximity.shape[0]):
                distance = distance_between_categories(cur_categories[i], cur_categories[j], data, categories)
                proximity[i][j] = distance
        print(proximity)
        category1 = cur_categories[int(np.where(proximity == np.max(proximity))[0][0])]
        category2 = cur_categories[int(np.where(proximity == np.max(proximity))[1][0])]
        print(str(category1) + " merge " + str(category2))
        # merge category1 and category2 into same class
        for i in range(0, len(categories)):
            if categories[i] == category2:
                categories[i] = category1
    
    # remark the label
    marks = list(set(categories))
    for i in range(0, len(marks)):
        categories = [x if x != marks[i] else i for x in categories]
    return categories

'''
i: line number
j: row number
data: dataset
categories: list of current cluster divided
'''
def distance_between_categories(i, j, data, categories):
    # get data by category
    category_i = [data[x] for x in range(0, len(categories)) if categories[x] == i]
    category_j = [data[x] for x in range(0, len(categories)) if categories[x] == j]
    
    min = dot_product(category_i[0], category_j[0])
    for x in category_i:
        for y in category_j:
            distance = dot_product(x, y)
            if min > distance:
                min = distance
    return min